"""
Flask web application for hematological risk prediction.

This application provides a web interface to predict:
- Anemia risk
- Neutropenia risk
- Thrombocytopenia risk

Uses the trained machine learning models saved in the models/ directory.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Global variables to store loaded models
models = {}
encoders = {}
scalers = {}

def load_models():
    """Load all trained models and associated files."""
    global models, encoders, scalers
    
    model_types = ['anemia', 'neutropenia', 'thrombocytopenia']
    
    for model_type in model_types:
        model_path = f'models/{model_type}_risk_model_latest.pkl'
        encoder_path = f'models/{model_type}_risk_encoder_latest.pkl'
        scaler_path = f'models/{model_type}_risk_scaler_latest.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        models[model_type] = joblib.load(model_path)
        encoders[model_type] = joblib.load(encoder_path)
        
        # Scaler is optional (only for Logistic Regression)
        if os.path.exists(scaler_path):
            scalers[model_type] = joblib.load(scaler_path)
        elif
            
        else:
            scalers[model_type] = None
    
    print("All models loaded successfully!")


def calculate_anc(wbc, neutrophils_percent):
    """
    Calculate Absolute Neutrophil Count (ANC).
    
    Parameters:
    -----------
    wbc : float
        White Blood Cell count (cells/μL or thousands/μL)
    neutrophils_percent : float
        Neutrophil percentage (0-100)
    
    Returns:
    --------
    float
        Absolute Neutrophil Count in thousands/μL
    """
    # Convert to cells/μL if needed
    if wbc < 1000:
        wbc_cells = wbc * 1000
    else:
        wbc_cells = wbc
    
    # Calculate ANC in cells/μL
    anc_cells = wbc_cells * (neutrophils_percent / 100)
    
    # Convert to thousands/μL for model input
    anc_thousands = anc_cells / 1000
    
    return anc_thousands


def normalize_features(rbc, wbc, platelets, anc=None):
    """
    Normalize features to match training data format.
    Converts all values to thousands/μL for consistency.
    
    Parameters:
    -----------
    rbc : float
        Red Blood Cell count (millions/μL)
    wbc : float
        White Blood Cell count (cells/μL or thousands/μL)
    platelets : float
        Platelet count (cells/μL or thousands/μL)
    anc : float, optional
        Absolute Neutrophil Count (cells/μL or thousands/μL)
    
    Returns:
    --------
    dict
        Dictionary with normalized features for each model
    """
    # Convert to thousands/μL for consistency with training data
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
    
    Parameters:
    -----------
    rbc : float
        Red Blood Cell count (millions/μL)
    wbc : float
        White Blood Cell count (cells/μL or thousands/μL)
    platelets : float
        Platelet count (cells/μL or thousands/μL)
    anc : float, optional
        Absolute Neutrophil Count (cells/μL or thousands/μL)
    neutrophils_percent : float, optional
        Neutrophil percentage (0-100). Used to calculate ANC if ANC not provided.
    
    Returns:
    --------
    dict
        Dictionary with predictions and probabilities for each risk type
    """
    # Check if models are loaded
    if not models or 'anemia' not in models:
        raise ValueError("Models not loaded. Please ensure models are loaded before making predictions.")
    
    # Calculate ANC if needed
    calculated_anc = None
    if anc is None and neutrophils_percent is not None:
        calculated_anc = calculate_anc(wbc, neutrophils_percent)
        anc = calculated_anc * 1000  # Convert back to cells/μL for display
    
    # Normalize features
    features = normalize_features(rbc, wbc, platelets, anc)
    
    predictions = {}
    
    # Predict Anemia Risk
    if 'anemia' not in models or 'anemia' not in encoders:
        raise KeyError("Anemia model or encoder not found in loaded models.")
    
    anemia_model = models['anemia']
    anemia_features = features['anemia']
    anemia_pred = anemia_model.predict(anemia_features)[0]
    anemia_risk = encoders['anemia'].inverse_transform([anemia_pred])[0]
    
    # Get prediction probabilities if available
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


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        # Check if models are loaded, try to load them if not
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
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        # Extract input values
        rbc = float(data.get('rbc', 0))
        wbc = float(data.get('wbc', 0))
        platelets = float(data.get('platelets', 0))
        anc = data.get('anc')
        anc = float(anc) if anc else None
        neutrophils_percent = data.get('neutrophils_percent')
        neutrophils_percent = float(neutrophils_percent) if neutrophils_percent else None
        
        # Validate required inputs
        if not all([rbc, wbc, platelets]):
            return jsonify({
                'error': 'RBC, WBC, and Platelets are required'
            }), 400
        
        # Make predictions
        predictions = predict_risk(rbc, wbc, platelets, anc, neutrophils_percent)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_values': {
                'rbc': rbc,
                'wbc': wbc,
                'platelets': platelets,
                'anc': anc,
                'neutrophils_percent': neutrophils_percent
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'loaded_models': list(models.keys()) if models else [],
        'encoders_loaded': list(encoders.keys()) if encoders else []
    })


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


if __name__ == '__main__':
    # Load models on startup
    try:
        load_models()
        print("All models loaded successfully!")
        print(f"Loaded models: {list(models.keys())}")
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        print("Please ensure all model files exist in the models/ directory.")
        print("Server will NOT start without models loaded.")
        import sys
        sys.exit(1)


