"""
ChemoGuard - Enhanced Flask web application for hematological risk prediction.

Features:
- User authentication and management
- Medical report upload and OCR extraction
- Toxicity risk prediction (Anemia, Neutropenia, Thrombocytopenia)
- Database storage (PostgreSQL)
- PDF export and sharing
- Prediction history tracking
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_login import LoginManager, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from config import config
from database import db, User
from auth import auth_bp
from routes import routes_bp
from cbc import ImplausibleValueError, panel_from_mapping
from clinical_flags import flags_for
from model_contract import (
    ContractError,
    check_estimator,
    contract_for,
    features_for,
    model_fingerprint,
)
from units import UnitError
import prediction_log
import joblib
import numpy as np
import os

def _env_flag(name, default=False):
    """Read a boolean from the environment. Anything unset is False."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config[os.environ.get('FLASK_CONFIG', 'development')])
# Single source of truth for debug. DevelopmentConfig sets DEBUG = True, so
# without this line the Werkzeug console is enabled even when app.run() is
# called with debug=False. Never on unless FLASK_DEBUG explicitly says so.
app.config['DEBUG'] = _env_flag('FLASK_DEBUG')

# Initialize extensions
csrf = CSRFProtect(app)
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Global variables to store loaded models
models = {}
encoders = {}
scalers = {}
# Content hash per loaded estimator, recorded at load time. The artefacts
# carry no version of their own, so the digest of the file that was actually
# read is the only identifier that cannot drift from what is serving.
model_versions = {}


@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login."""
    return User.query.get(int(user_id))


MODEL_TYPES = ['anemia', 'neutropenia', 'thrombocytopenia']


def load_models():
    """Load all trained models and associated files."""
    global models, encoders, scalers

    # Clear existing models
    models.clear()
    encoders.clear()
    scalers.clear()
    model_versions.clear()

    model_types = MODEL_TYPES
    
    # Get the base directory (where app.py is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for model_type in model_types:
        model_path = os.path.join(base_dir, f'models/{model_type}_risk_model_latest.pkl')
        encoder_path = os.path.join(base_dir, f'models/{model_type}_risk_encoder_latest.pkl')
        scaler_path = os.path.join(base_dir, f'models/{model_type}_risk_scaler_latest.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            estimator = joblib.load(model_path)
            # Refuse an estimator whose arity disagrees with the units and
            # feature order declared in model_contract. An artefact that does
            # not match its declared contract is exactly the Finding 7
            # failure, and it must not reach .predict().
            check_estimator(model_type, estimator)
            models[model_type] = estimator
            encoders[model_type] = joblib.load(encoder_path)
            model_versions[model_type] = model_fingerprint(model_path)
        except ContractError:
            raise
        except Exception as e:
            raise Exception(f"Error loading {model_type} model: {str(e)}")
        # ASCII only, and outside the try. The Windows console defaults to
        # cp1252, where a non-ASCII character raises UnicodeEncodeError; when
        # that print sat inside the try, the encoding error was caught and
        # re-raised as "Error loading <model> model", hiding the real cause.
        print(f"[ok] Loaded {model_type} model and encoder")
        
        # Scaler is optional (only for Logistic Regression)
        if os.path.exists(scaler_path):
            scalers[model_type] = joblib.load(scaler_path)
        else:
            scalers[model_type] = None
    
    print(f"All models loaded successfully! ({len(models)} models)")


# Load models when module is imported (works for both python app.py and flask run)
# This ensures models are loaded regardless of how the app is started
def initialize_models():
    """Initialize models on app startup."""
    prediction_log.configure_default_handler()
    try:
        load_models()
        print("All models loaded successfully on import!")
        print(f"Loaded models: {list(models.keys())}")
        return True
    except Exception as e:
        print(f"WARNING: Failed to load models on import: {e}")
        print("Models will need to be loaded manually or server restarted.")
        print("You can reload models by calling: POST /reload_models")
        return False

# Try to load models on import
_models_loaded = initialize_models()


class ModelsUnavailable(RuntimeError):
    """No estimator could be served. Mapped to HTTP 503 by the endpoints."""


def _ensure_models():
    if models and 'anemia' in models:
        return
    try:
        print("Models not loaded, attempting to load now...")
        load_models()
        print("Models loaded successfully!")
    except Exception as load_error:
        raise ModelsUnavailable(
            f"Models not loaded. Failed to load: {load_error}"
        ) from load_error


def _predict_one(model_type, panel):
    """Run one estimator against a panel of Quantities.

    The feature row comes from model_contract.features_for, which is the only
    place a Quantity becomes a float for a model, and which converts to the
    unit that model's contract declares. Nothing here inspects a magnitude.
    """
    estimator = models[model_type]
    encoder = encoders[model_type]

    row, record = features_for(model_type, panel)
    features = np.array([row])

    predicted = estimator.predict(features)[0]
    risk = encoder.inverse_transform([predicted])[0]

    if hasattr(estimator, 'predict_proba'):
        proba = estimator.predict_proba(features)[0]
        probabilities = {str(k): float(v) for k, v in zip(encoder.classes_, proba)}
    else:
        probabilities = None

    return (
        {
            'risk': str(risk),
            'probabilities': probabilities,
            'model_version': model_versions.get(model_type),
            'feature_units': contract_for(model_type).as_dict()['features'],
        },
        record,
    )


def predict_risk(panel, request_id=None, user_id=None, report_id=None):
    """Predict all three risks from a panel of Quantities.

    Takes a CBCPanel, never bare floats. Every value in the panel arrived
    with a unit and has been converted to its analyte's canonical unit
    exactly once; this function converts again only through the declared
    model contract, immediately before .predict().

    Returns (predictions, flags, feature_records). Emits one structured log
    record per call.
    """
    _ensure_models()

    panel = panel.with_derived_anc()
    request_id = request_id or prediction_log.new_prediction_id()

    predictions = {}
    feature_records = {}
    used_versions = {}

    for model_type in ('anemia', 'thrombocytopenia'):
        if model_type not in models or model_type not in encoders:
            raise ModelsUnavailable(
                f"{model_type} model or encoder not found in loaded models."
            )
        predictions[model_type], feature_records[model_type] = _predict_one(
            model_type, panel
        )
        used_versions[model_type] = model_versions.get(model_type)

    # Neutropenia needs ANC, which is either reported or derived from WBC and
    # the neutrophil percentage. Absent both, it is not predicted -- and that
    # is reported as an absence, not as a Low.
    if panel.get('anc') is not None:
        if 'neutropenia' not in models or 'neutropenia' not in encoders:
            raise ModelsUnavailable(
                "neutropenia model or encoder not found in loaded models."
            )
        predictions['neutropenia'], feature_records['neutropenia'] = _predict_one(
            'neutropenia', panel
        )
        predictions['neutropenia']['anc_used'] = panel.get('anc').as_dict()
        predictions['neutropenia']['anc_source'] = panel.unit_provenance.get('anc')
        used_versions['neutropenia'] = model_versions.get('neutropenia')
    else:
        predictions['neutropenia'] = {
            'risk': None,
            'probabilities': None,
            'error': (
                'ANC is required for neutropenia prediction, and neither an '
                'ANC nor a neutrophil percentage was supplied.'
            ),
        }

    flags = flags_for(panel)

    prediction_log.log_prediction(
        request_id=request_id,
        panel=panel,
        predictions=predictions,
        feature_records=feature_records,
        model_versions=used_versions,
        flags=flags,
        user_id=user_id,
        report_id=report_id,
    )

    return predictions, flags, feature_records


RISK_SCORES = {'Low': 1, 'Medium': 2, 'High': 3}


def combine_severity(predictions):
    """Worst risk across the models that actually produced one.

    A model that did not run is reported as not run. The code this replaces
    used `risk_scores.get(..., 1)`, which scored an absent neutropenia
    prediction as Low -- the same fail-toward-normal direction as the unit
    inference, and equally silent.
    """
    scored = {
        name: RISK_SCORES[body['risk']]
        for name, body in predictions.items()
        if body.get('risk') in RISK_SCORES
    }
    not_predicted = sorted(set(predictions) - set(scored))

    if not scored:
        return {
            'severity': None,
            'score': None,
            'basis': [],
            'not_predicted': not_predicted,
        }

    max_score = max(scored.values())
    return {
        'severity': {1: 'Low', 2: 'Moderate', 3: 'High'}[max_score],
        'score': max_score,
        'basis': sorted(scored),
        'not_predicted': not_predicted,
    }


def prediction_error_response(exc, request_id, raw_input, user_id=None):
    """Map a prediction failure to a response, and log the refusal.

    422 rather than 400 for a unit problem: the request was well formed and
    we are declining to interpret it, which is a different thing from a
    field being absent and worth being able to tell apart in the logs.
    """
    if isinstance(exc, UnitError):
        reason, status = 'missing_or_unknown_unit', 422
    elif isinstance(exc, ImplausibleValueError):
        reason, status = 'implausible_value', 422
    elif isinstance(exc, ModelsUnavailable):
        reason, status = 'models_unavailable', 503
    elif isinstance(exc, ContractError):
        reason, status = 'model_contract_violation', 503
    else:
        return None

    prediction_log.log_refusal(
        request_id=request_id,
        reason=reason,
        detail=str(exc),
        raw_input=raw_input,
        user_id=user_id,
    )
    return jsonify({
        'success': False,
        'error': str(exc),
        'reason': reason,
        'request_id': request_id,
    }), status


# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(routes_bp, url_prefix='/api')


# Legacy endpoint (for backward compatibility)
@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy predict endpoint (backward compatibility).

    Backward compatible in route and response shape only. It no longer
    accepts a bare number: every value needs a unit, in one of the three
    shapes cbc.panel_from_mapping documents. A caller sending
    {"wbc": 5800} now gets 422 instead of a prediction computed from a
    guess.
    """
    request_id = prediction_log.new_prediction_id()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    try:
        panel = panel_from_mapping(data)
        missing = panel.missing_required()
        if missing:
            return jsonify({
                'error': 'RBC, WBC, and Platelets are required',
                'missing': missing,
                'request_id': request_id,
            }), 400

        predictions, flags, _ = predict_risk(panel, request_id=request_id)
    except Exception as exc:
        handled = prediction_error_response(exc, request_id, data)
        if handled is not None:
            return handled
        return jsonify({'error': str(exc), 'request_id': request_id}), 500

    combined = combine_severity(predictions)
    return jsonify({
        'success': True,
        'request_id': request_id,
        'predictions': predictions,
        'combined_severity': combined['severity'],
        'combined_toxicity': combined['severity'],  # Alias for clarity
        'combined': combined,
        'flags': [f.as_dict() for f in flags],
        'input_values': panel.as_dict(),
    })


# Export and sharing routes
@app.route('/api/export/pdf/<int:prediction_id>', methods=['GET'])
@login_required
def export_pdf(prediction_id):
    """Export prediction as PDF."""
    from export import generate_pdf_report
    
    try:
        pdf_buffer = generate_pdf_report(prediction_id, current_user)
        filename = f'chemoguard_report_{prediction_id}.pdf'
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/share/qr/<int:prediction_id>', methods=['GET'])
@login_required
def get_qr_code(prediction_id):
    """Generate QR code for sharing prediction."""
    from export import generate_qr_code
    
    try:
        base_url = request.host_url.rstrip('/')
        qr_buffer = generate_qr_code(prediction_id, base_url)
        return send_file(
            qr_buffer,
            mimetype='image/png',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check.

    Reports the absence of models rather than hiding it. No models is the
    expected state until the NHANES models land: the previous artefacts were
    quarantined because they were trained on a withdrawn dataset with
    rule-derived labels (CLAUDE.md Findings 1 and 5).

    Always returns 200 -- the service itself is up. Read `models_loaded`;
    do not infer availability from the status code.
    """
    missing = [name for name in MODEL_TYPES if name not in models]
    payload = {
        'status': 'ok' if not missing else 'degraded',
        'models_loaded': not missing,
        'loaded_models': sorted(models),
        'encoders_loaded': sorted(encoders),
        'expected_models': list(MODEL_TYPES),
        'missing_models': missing,
        # The unit each estimator is fed, and the digest of the artefact
        # actually serving. Finding 7's defect was that neither was recorded
        # anywhere; exposing both makes a mismatch observable from outside.
        'model_versions': dict(model_versions),
        'feature_contracts': {
            name: contract_for(name).as_dict()['features'] for name in MODEL_TYPES
        },
    }
    if missing:
        payload['detail'] = (
            'Risk prediction is unavailable. The previous models were '
            'quarantined: they were trained on a dataset withdrawn from '
            'Mendeley, against labels that were threshold rules over their '
            'own input features. Awaiting the NHANES-trained replacements.'
        )
    return jsonify(payload)


@app.route('/reload_models', methods=['POST'])
@login_required
def reload_models():
    """Reload every model artefact from disk.

    Authenticated because it mutates process state. CSRF protection alone was
    not access control here: any anonymous client that fetches ``/`` is handed
    a token, so the endpoint was reachable by anyone who asked for one. It was
    low-impact only while ``models/`` stayed empty, which stops being true the
    moment the NHANES artefacts land.
    """
    try:
        load_models()
        return jsonify({
            'success': True,
            'message': 'Models reloaded successfully',
            'loaded_models': list(models.keys())
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Make predict_risk available to routes module
import sys
sys.modules['app'] = sys.modules[__name__]


if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        print("Database tables created/verified.")
    
    # Load models on startup. Their absence no longer aborts startup: since
    # the quarantine there are no models to load, and that is the expected
    # state until the NHANES models land. The app serves, /health reports
    # the gap, and the prediction endpoints refuse with 503 rather than
    # guessing.
    try:
        load_models()
        print(f"Loaded models: {sorted(models)}")
    except Exception as e:
        print(f"NOTICE: starting without models ({e})")
        print("Risk prediction will return 503. /health reports models_loaded: false.")

    print("Starting Flask application...")
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)

