"""
Database models and configuration for ChemoGuard application.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()


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
    
    # Extracted hematological parameters
    hb = db.Column(db.Float)  # Hemoglobin
    rbc = db.Column(db.Float)  # Red Blood Cells
    wbc = db.Column(db.Float)  # White Blood Cells
    anc = db.Column(db.Float)  # Absolute Neutrophil Count
    platelets = db.Column(db.Float)
    neutrophils_percent = db.Column(db.Float)
    hematocrit = db.Column(db.Float)
    mcv = db.Column(db.Float)  # Mean Corpuscular Volume
    mch = db.Column(db.Float)  # Mean Corpuscular Hemoglobin
    mchc = db.Column(db.Float)  # Mean Corpuscular Hemoglobin Concentration
    
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
    
    # Input values used for prediction
    input_rbc = db.Column(db.Float, nullable=False)
    input_wbc = db.Column(db.Float, nullable=False)
    input_platelets = db.Column(db.Float, nullable=False)
    input_anc = db.Column(db.Float, nullable=True)
    input_neutrophils_percent = db.Column(db.Float, nullable=True)
    
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

