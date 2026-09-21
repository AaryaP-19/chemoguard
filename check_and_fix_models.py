"""
Script to check if models can be loaded and diagnose issues.
Run this to verify your models are accessible before starting the server.
"""

import os
import joblib

def check_models():
    """Check if all required model files exist and can be loaded."""
    model_types = ['anemia', 'neutropenia', 'thrombocytopenia']
    issues = []
    
    print("Checking model files...")
    print("-" * 50)
    
    for model_type in model_types:
        print(f"\nChecking {model_type} model:")
        
        model_path = f'models/{model_type}_risk_model_latest.pkl'
        encoder_path = f'models/{model_type}_risk_encoder_latest.pkl'
        scaler_path = f'models/{model_type}_risk_scaler_latest.pkl'
        
        # Check model file
        if not os.path.exists(model_path):
            print(f"  ❌ Model file NOT FOUND: {model_path}")
            issues.append(f"Missing model: {model_path}")
        else:
            print(f"  ✅ Model file exists: {model_path}")
            try:
                model = joblib.load(model_path)
                print(f"     Model type: {type(model).__name__}")
            except Exception as e:
                print(f"  ❌ Error loading model: {e}")
                issues.append(f"Error loading {model_path}: {e}")
        
        # Check encoder file
        if not os.path.exists(encoder_path):
            print(f"  ❌ Encoder file NOT FOUND: {encoder_path}")
            issues.append(f"Missing encoder: {encoder_path}")
        else:
            print(f"  ✅ Encoder file exists: {encoder_path}")
            try:
                encoder = joblib.load(encoder_path)
                print(f"     Encoder classes: {encoder.classes_ if hasattr(encoder, 'classes_') else 'N/A'}")
            except Exception as e:
                print(f"  ❌ Error loading encoder: {e}")
                issues.append(f"Error loading {encoder_path}: {e}")
        
        # Check scaler file (optional)
        if os.path.exists(scaler_path):
            print(f"  ✅ Scaler file exists: {scaler_path}")
            try:
                scaler = joblib.load(scaler_path)
                print(f"     Scaler type: {type(scaler).__name__}")
            except Exception as e:
                print(f"  ⚠️  Error loading scaler (optional): {e}")
        else:
            print(f"  ℹ️  Scaler file not found (optional): {scaler_path}")
    
    print("\n" + "-" * 50)
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before starting the server.")
        return False
    else:
        print("\n✅ All models are accessible and can be loaded!")
        print("You can now start the Flask server with: python app.py")
        return True

if __name__ == '__main__':
    # Change to script directory
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    check_models()

