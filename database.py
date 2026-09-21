"""
Database models and configuration for ChemoGuard application.

Every stored laboratory value lives in a column named for the unit it holds
-- ``wbc_cells_per_ul``, not ``wbc``. The old columns were bare Floats whose
scale depended on which extraction path had written them, and renaming
rather than reinterpreting is deliberate: a row written under the old schema
now fails to load instead of being read 1000x wrong. CLAUDE.md Finding 3 is
exactly a 1000x unit change made without a rename, and it took an audit to
find. See MIGRATION_UNITS.md.

Alongside the canonical columns, ``reported_quantities`` keeps what the
source actually said, per analyte, unconverted -- so the original reading
survives even if the canonical unit is ever redefined.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

from cbc import ANALYTES, CBCPanel, build_panel

db = SQLAlchemy()

#: Analyte -> the column suffix naming its canonical unit. Derived from the
#: canonical units declared in cbc.ANALYTES so the two cannot drift.
_UNIT_SUFFIX = {
    "cells/uL": "cells_per_ul",
    "thousand/uL": "thousand_per_ul",
    "million/uL": "million_per_ul",
    "g/dL": "g_per_dl",
    "g/L": "g_per_l",
    "%": "percent",
    "fL": "fl",
    "pg": "pg",
}


def column_name(analyte_key, prefix=""):
    """The column that holds an analyte's canonical value.

    ``column_name("wbc")`` -> ``wbc_cells_per_ul``
    ``column_name("wbc", "input_")`` -> ``input_wbc_cells_per_ul``
    """
    analyte = ANALYTES[analyte_key]
    suffix = _UNIT_SUFFIX[analyte.canonical.symbol]
    stem = analyte_key
    # neutrophils_percent already names its unit; don't say it twice.
    if stem.endswith("_" + suffix):
        return prefix + stem
    return "{0}{1}_{2}".format(prefix, stem, suffix)


#: Analytes a MedicalReport stores, and those a Prediction stores as inputs.
REPORT_ANALYTES = tuple(ANALYTES)
PREDICTION_INPUT_ANALYTES = ("rbc", "wbc", "platelets", "anc", "neutrophils_percent")


def _columns_for(panel, analytes, prefix=""):
    values = {}
    for key in analytes:
        quantity = panel.get(key)
        values[column_name(key, prefix)] = None if quantity is None else quantity.value
    return values


def report_columns(panel):
    """Canonical column values for a MedicalReport."""
    return _columns_for(panel, REPORT_ANALYTES)


def prediction_input_columns(panel):
    """Canonical column values for a Prediction's inputs."""
    return _columns_for(panel, PREDICTION_INPUT_ANALYTES, prefix="input_")


def reported_quantities(panel):
    """The as-recorded values and units, for the JSON provenance column."""
    return {
        key: {
            "value": quantity.value,
            "unit": quantity.unit.symbol,
            "unit_source": panel.unit_provenance.get(key, "unknown"),
        }
        for key, quantity in panel.reported.items()
    }


def model_feature_units(predictions):
    """Which units each estimator was fed, taken from the prediction bodies.

    Persisting this is the direct answer to CLAUDE.md Finding 7: the scale a
    model expected was nowhere in the saved state, so a stored prediction
    could not be checked against the artefact that produced it.
    """
    return {
        name: body.get("feature_units")
        for name, body in predictions.items()
        if body.get("feature_units")
    }


def panel_from_columns(row, analytes, prefix=""):
    """Rebuild a CBCPanel from stored canonical columns.

    Reads the canonical column and pairs it with the canonical unit declared
    in cbc.ANALYTES. The reported originals, where present, come back from
    ``reported_quantities``.
    """
    values = {}
    for key in analytes:
        value = getattr(row, column_name(key, prefix), None)
        if value is not None:
            values[key] = (value, ANALYTES[key].canonical)
    return build_panel(values, unit_source="stored")


class User(UserMixin, db.Model):
    """User model for authentication and profile management."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reports = db.relationship('MedicalReport', backref='user', lazy=True, cascade='all, delete-orphan')
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash."""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class MedicalReport(db.Model):
    """Medical report model for storing uploaded reports and extracted data."""
    __tablename__ = 'medical_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(50))  # PDF, JPEG, PNG
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Extracted haematological parameters, canonical values only. The column
    # name states the unit; there is no other convention to remember and no
    # way to store one of these without saying what it means.
    hb_g_per_dl = db.Column(db.Float)                 # Haemoglobin
    rbc_million_per_ul = db.Column(db.Float)          # Red Blood Cells
    wbc_cells_per_ul = db.Column(db.Float)            # White Blood Cells
    anc_cells_per_ul = db.Column(db.Float)            # Absolute Neutrophil Count
    platelets_cells_per_ul = db.Column(db.Float)
    neutrophils_percent = db.Column(db.Float)
    hematocrit_percent = db.Column(db.Float)
    mcv_fl = db.Column(db.Float)                      # Mean Corpuscular Volume
    mch_pg = db.Column(db.Float)                      # Mean Corpuscular Haemoglobin
    mchc_g_per_dl = db.Column(db.Float)               # MCH Concentration

    #: What the report itself said, per analyte: value, unit, and whether the
    #: unit was read off the document or supplied by an operator. Kept so the
    #: original reading survives a later change to the canonical unit, and so
    #: an extraction can be audited against the source image.
    reported_quantities = db.Column(db.JSON)

    # OCR extraction metadata
    ocr_extracted = db.Column(db.Boolean, default=False)
    ocr_confidence = db.Column(db.Float)
    extraction_date = db.Column(db.DateTime)
    raw_ocr_text = db.Column(db.Text)  # Store raw OCR output for debugging
    
    # Relationships
    predictions = db.relationship('Prediction', backref='report', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<MedicalReport {self.filename}>'


class Prediction(db.Model):
    """Prediction model for storing toxicity risk predictions and history."""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    report_id = db.Column(db.Integer, db.ForeignKey('medical_reports.id'), nullable=True)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    version = db.Column(db.Integer, default=1)  # For versioning history
    #: Correlates this row with its entry in the structured prediction log.
    request_id = db.Column(db.String(32), index=True)

    # Input values used for prediction, canonical, unit in the column name.
    input_rbc_million_per_ul = db.Column(db.Float, nullable=False)
    input_wbc_cells_per_ul = db.Column(db.Float, nullable=False)
    input_platelets_cells_per_ul = db.Column(db.Float, nullable=False)
    input_anc_cells_per_ul = db.Column(db.Float, nullable=True)
    input_neutrophils_percent = db.Column(db.Float, nullable=True)

    #: The inputs exactly as the report or operator gave them, with units.
    input_reported_quantities = db.Column(db.JSON)
    #: The unit each estimator was actually fed, per model. Without this a
    #: stored prediction cannot be checked against the artefact that made it
    #: -- CLAUDE.md Finding 7, where the scale a model expected was recorded
    #: nowhere in the saved state.
    model_feature_units = db.Column(db.JSON)
    
    # Prediction results
    anemia_risk = db.Column(db.String(20))  # Low, Medium, High
    anemia_probability_low = db.Column(db.Float)
    anemia_probability_medium = db.Column(db.Float)
    anemia_probability_high = db.Column(db.Float)
    
    neutropenia_risk = db.Column(db.String(20))
    neutropenia_probability_low = db.Column(db.Float)
    neutropenia_probability_medium = db.Column(db.Float)
    neutropenia_probability_high = db.Column(db.Float)
    
    thrombocytopenia_risk = db.Column(db.String(20))
    thrombocytopenia_probability_low = db.Column(db.Float)
    thrombocytopenia_probability_medium = db.Column(db.Float)
    thrombocytopenia_probability_high = db.Column(db.Float)
    
    # Combined toxicity severity
    combined_severity = db.Column(db.String(20))  # Low, Moderate, High
    combined_score = db.Column(db.Float)  # Calculated combined risk score
    
    # Additional metadata
    notes = db.Column(db.Text)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.combined_severity}>'


class PredictionHistory(db.Model):
    """Versioned history of predictions for tracking changes over time."""
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    version = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Store full prediction data as JSON for history
    prediction_data = db.Column(db.JSON)
    change_summary = db.Column(db.Text)  # What changed in this version
    
    def __repr__(self):
        return f'<PredictionHistory {self.prediction_id} v{self.version}>'

