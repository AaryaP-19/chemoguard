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
            models[model_type] = joblib.load(model_path)
            encoders[model_type] = joblib.load(encoder_path)
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


def calculate_anc(wbc, neutrophils_percent):
    """Calculate Absolute Neutrophil Count (ANC)."""
    if wbc < 1000:
        wbc_cells = wbc * 1000
    else:
        wbc_cells = wbc
    
    anc_cells = wbc_cells * (neutrophils_percent / 100)
    anc_thousands = anc_cells / 1000
    
    return anc_thousands


def normalize_features(rbc, wbc, platelets, anc=None):
    """Normalize features to match training data format."""
    if wbc > 1000:
        wbc_thousands = wbc / 1000
    else:
        wbc_thousands = wbc
    
    if platelets > 1000:
        platelets_thousands = platelets / 1000
    else:
        platelets_thousands = platelets
    
    if anc is not None:
        if anc > 1000:
            anc_thousands = anc / 1000
        else:
            anc_thousands = anc
    else:
        anc_thousands = None
    
    features = {
        'anemia': np.array([[rbc, wbc_thousands, platelets_thousands]]),
        'thrombocytopenia': np.array([[rbc, wbc_thousands, platelets_thousands]]),
        'neutropenia': np.array([[rbc, wbc_thousands, platelets_thousands, anc_thousands]]) if anc_thousands is not None else None
    }
    
    return features


def predict_risk(rbc, wbc, platelets, anc=None, neutrophils_percent=None):
    """
    Make predictions for all three risk types.
    This function is used by both the old /predict endpoint and the new routes.
    """
    # Check if models are loaded, try to load them if not
    if not models or 'anemia' not in models:
        try:
            print("Models not loaded, attempting to load now...")
            load_models()
            print("Models loaded successfully!")
        except Exception as load_error:
            raise ValueError(f"Models not loaded. Failed to load: {str(load_error)}")
    
    calculated_anc = None
    if anc is None and neutrophils_percent is not None:
        calculated_anc = calculate_anc(wbc, neutrophils_percent)
        anc = calculated_anc * 1000
    
    features = normalize_features(rbc, wbc, platelets, anc)
    predictions = {}
    
    # Predict Anemia Risk
    if 'anemia' not in models or 'anemia' not in encoders:
        raise KeyError("Anemia model or encoder not found in loaded models.")
    
    anemia_model = models['anemia']
    anemia_features = features['anemia']
    anemia_pred = anemia_model.predict(anemia_features)[0]
    anemia_risk = encoders['anemia'].inverse_transform([anemia_pred])[0]
    
    if hasattr(anemia_model, 'predict_proba'):
        anemia_proba = anemia_model.predict_proba(anemia_features)[0]
        anemia_proba_dict = dict(zip(encoders['anemia'].classes_, anemia_proba))
    else:
        anemia_proba_dict = None
    
    predictions['anemia'] = {
        'risk': anemia_risk,
        'probabilities': anemia_proba_dict
    }
    
    # Predict Thrombocytopenia Risk
    if 'thrombocytopenia' not in models or 'thrombocytopenia' not in encoders:
        raise KeyError("Thrombocytopenia model or encoder not found in loaded models.")
    
    thrombocytopenia_model = models['thrombocytopenia']
    thrombocytopenia_features = features['thrombocytopenia']
    thrombocytopenia_pred = thrombocytopenia_model.predict(thrombocytopenia_features)[0]
    thrombocytopenia_risk = encoders['thrombocytopenia'].inverse_transform([thrombocytopenia_pred])[0]
    
    if hasattr(thrombocytopenia_model, 'predict_proba'):
        thrombocytopenia_proba = thrombocytopenia_model.predict_proba(thrombocytopenia_features)[0]
        thrombocytopenia_proba_dict = dict(zip(encoders['thrombocytopenia'].classes_, thrombocytopenia_proba))
    else:
        thrombocytopenia_proba_dict = None
    
    predictions['thrombocytopenia'] = {
        'risk': thrombocytopenia_risk,
        'probabilities': thrombocytopenia_proba_dict
    }
    
    # Predict Neutropenia Risk (if ANC available)
    if features['neutropenia'] is not None:
        if 'neutropenia' not in models or 'neutropenia' not in encoders:
            raise KeyError("Neutropenia model or encoder not found in loaded models.")
        
        neutropenia_model = models['neutropenia']
        neutropenia_features = features['neutropenia']
        neutropenia_pred = neutropenia_model.predict(neutropenia_features)[0]
        neutropenia_risk = encoders['neutropenia'].inverse_transform([neutropenia_pred])[0]
        
        if hasattr(neutropenia_model, 'predict_proba'):
            neutropenia_proba = neutropenia_model.predict_proba(neutropenia_features)[0]
            neutropenia_proba_dict = dict(zip(encoders['neutropenia'].classes_, neutropenia_proba))
        else:
            neutropenia_proba_dict = None
        
        predictions['neutropenia'] = {
            'risk': neutropenia_risk,
            'probabilities': neutropenia_proba_dict,
            'anc_used': anc if anc else (calculated_anc * 1000 if calculated_anc else None)
        }
    else:
        predictions['neutropenia'] = {
            'risk': 'ANC required',
            'probabilities': None,
            'error': 'ANC or Neutrophil percentage is required for neutropenia prediction'
        }
    
    return predictions


# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(routes_bp, url_prefix='/api')


# Legacy endpoint (for backward compatibility)
@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy predict endpoint (backward compatibility)."""
    try:
        # Ensure models are loaded
        if not models or len(models) == 0:
            try:
                print("Models not loaded, attempting to load now...")
                load_models()
                print("Models loaded successfully!")
            except Exception as load_error:
                return jsonify({
                    'error': f'Models not loaded. Failed to load: {str(load_error)}'
                }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        rbc = float(data.get('rbc', 0))
        wbc = float(data.get('wbc', 0))
        platelets = float(data.get('platelets', 0))
        anc = data.get('anc')
        anc = float(anc) if anc else None
        neutrophils_percent = data.get('neutrophils_percent')
        neutrophils_percent = float(neutrophils_percent) if neutrophils_percent else None
        
        if not all([rbc, wbc, platelets]):
            return jsonify({'error': 'RBC, WBC, and Platelets are required'}), 400
        
        predictions = predict_risk(rbc, wbc, platelets, anc, neutrophils_percent)
        
        # Calculate combined severity for legacy endpoint
        risk_scores = {'Low': 1, 'Medium': 2, 'High': 3}
        anemia_score = risk_scores.get(predictions['anemia']['risk'], 1)
        neutropenia_score = risk_scores.get(predictions.get('neutropenia', {}).get('risk', 'Low'), 1)
        thrombocytopenia_score = risk_scores.get(predictions['thrombocytopenia']['risk'], 1)
        max_score = max(anemia_score, neutropenia_score, thrombocytopenia_score)
        combined_severity = {1: 'Low', 2: 'Moderate', 3: 'High'}[max_score]
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'combined_severity': combined_severity,
            'combined_toxicity': combined_severity,  # Alias for clarity
            'input_values': {
                'rbc': rbc,
                'wbc': wbc,
                'platelets': platelets,
                'anc': anc,
                'neutrophils_percent': neutrophils_percent
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
def reload_models():
    """Reload models endpoint."""
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

