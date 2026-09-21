"""
Script to predict hematological risks using trained models.

This script loads the trained models and makes predictions for:
1. Anemia risk
2. Neutropenia risk (requires ANC calculation)
3. Thrombocytopenia risk

Usage:
    python predict_risk.py --rbc 4.5 --wbc 8.0 --platelets 250 --anc 5.0
    or
    python predict_risk.py --rbc 4.5 --wbc 8000 --platelets 250000 --anc 5000 --calculate_anc
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path


def load_models():
    """Load all trained models and associated files."""
    models = {}
    
    for risk_type in ['anemia', 'neutropenia', 'thrombocytopenia']:
        model_path = f'models/{risk_type}_risk_model_latest.pkl'
        encoder_path = f'models/{risk_type}_risk_encoder_latest.pkl'
        scaler_path = f'models/{risk_type}_risk_scaler_latest.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Please train models first.")
        
        models[risk_type] = {
            'model': joblib.load(model_path),
            'encoder': joblib.load(encoder_path),
            'scaler': joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        }
    
    return models


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
        Absolute Neutrophil Count
    """
    # Convert to cells/μL if needed
    if wbc < 1000:
        wbc_cells = wbc * 1000
    else:
        wbc_cells = wbc
    
    anc = wbc_cells * (neutrophils_percent / 100)
    return anc


def normalize_features(rbc, wbc, platelets, anc=None):
    """
    Normalize features to match training data format.
    
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


def predict_risk(models, rbc, wbc, platelets, anc=None, neutrophils_percent=None):
    """
    Make predictions for all three risk types.
    
    Parameters:
    -----------
    models : dict
        Dictionary of loaded models
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
        Dictionary with predictions for each risk type
    """
    # Calculate ANC if needed
    if anc is None and neutrophils_percent is not None:
        anc = calculate_anc(wbc, neutrophils_percent)
        print(f"Calculated ANC: {anc:.2f} cells/μL")
    
    # Normalize features
    features = normalize_features(rbc, wbc, platelets, anc)
    
    predictions = {}
    
    # Predict Anemia Risk
    anemia_model = models['anemia']['model']
    anemia_features = features['anemia']
    anemia_pred = anemia_model.predict(anemia_features)[0]
    anemia_risk = models['anemia']['encoder'].inverse_transform([anemia_pred])[0]
    predictions['anemia'] = anemia_risk
    
    # Predict Thrombocytopenia Risk
    thrombocytopenia_model = models['thrombocytopenia']['model']
    thrombocytopenia_features = features['thrombocytopenia']
    thrombocytopenia_pred = thrombocytopenia_model.predict(thrombocytopenia_features)[0]
    thrombocytopenia_risk = models['thrombocytopenia']['encoder'].inverse_transform([thrombocytopenia_pred])[0]
    predictions['thrombocytopenia'] = thrombocytopenia_risk
    
    # Predict Neutropenia Risk (if ANC available)
    if features['neutropenia'] is not None:
        neutropenia_model = models['neutropenia']['model']
        neutropenia_features = features['neutropenia']
        neutropenia_pred = neutropenia_model.predict(neutropenia_features)[0]
        neutropenia_risk = models['neutropenia']['encoder'].inverse_transform([neutropenia_pred])[0]
        predictions['neutropenia'] = neutropenia_risk
    else:
        predictions['neutropenia'] = "ANC required for prediction"
    
    return predictions


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Predict hematological risks using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using values in standard units (RBC in millions, WBC/Platelets in thousands)
  python predict_risk.py --rbc 4.5 --wbc 8.0 --platelets 250 --anc 5.0
  
  # Using values in cells/μL (will be converted automatically)
  python predict_risk.py --rbc 4.5 --wbc 8000 --platelets 250000 --anc 5000
  
  # Calculate ANC from WBC and neutrophil percentage
  python predict_risk.py --rbc 4.5 --wbc 8000 --platelets 250000 --neutrophils 60
        """
    )
    
    parser.add_argument('--rbc', type=float, required=True,
                       help='Red Blood Cell count (millions/μL)')
    parser.add_argument('--wbc', type=float, required=True,
                       help='White Blood Cell count (cells/μL or thousands/μL)')
    parser.add_argument('--platelets', type=float, required=True,
                       help='Platelet count (cells/μL or thousands/μL)')
    parser.add_argument('--anc', type=float, default=None,
                       help='Absolute Neutrophil Count (cells/μL or thousands/μL). Required for neutropenia prediction.')
    parser.add_argument('--neutrophils', type=float, default=None,
                       help='Neutrophil percentage (0-100). Used to calculate ANC if ANC not provided.')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HEMATOLOGICAL RISK PREDICTION")
    print("=" * 80)
    print(f"\nInput values:")
    print(f"  RBC: {args.rbc} millions/μL")
    print(f"  WBC: {args.wbc} {'cells/μL' if args.wbc > 1000 else 'thousands/μL'}")
    print(f"  Platelets: {args.platelets} {'cells/μL' if args.platelets > 1000 else 'thousands/μL'}")
    if args.anc is not None:
        print(f"  ANC: {args.anc} {'cells/μL' if args.anc > 1000 else 'thousands/μL'}")
    if args.neutrophils is not None:
        print(f"  Neutrophils: {args.neutrophils}%")
    
    print("\n" + "=" * 80)
    print("Loading models...")
    print("=" * 80)
    
    try:
        models = load_models()
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_risk_prediction_models.py first to train the models.")
        return
    
    print("\n" + "=" * 80)
    print("Making predictions...")
    print("=" * 80)
    
    predictions = predict_risk(
        models, 
        args.rbc, 
        args.wbc, 
        args.platelets, 
        anc=args.anc,
        neutrophils_percent=args.neutrophils
    )
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n  Anemia Risk:           {predictions['anemia']}")
    print(f"  Neutropenia Risk:       {predictions['neutropenia']}")
    print(f"  Thrombocytopenia Risk:  {predictions['thrombocytopenia']}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


