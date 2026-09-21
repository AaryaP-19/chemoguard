"""
Script to train three machine learning models for predicting:
1. Anemia risk (using RBC, WBC, Platelets)
2. Neutropenia risk (using RBC, WBC, Platelets, ANC)
3. Thrombocytopenia risk (using RBC, WBC, Platelets)

This script:
- Loads the three datasets with risk labels
- Prepares features and targets
- Trains multiple ML models for each task
- Evaluates models using cross-validation
- Saves the trained models
- Generates evaluation reports
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from datetime import datetime

# Try importing XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


def load_and_prepare_data():
    """Load all three datasets and prepare features."""
    print("=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    
    # Load datasets
    anemia_df = pd.read_csv("cbc_with_anemia_risk.csv")
    neutropenia_df = pd.read_csv("cbc_with_neutropenia_risk.csv")
    thrombocytopenia_df = pd.read_csv("cbc_with_thrombocytopenia_risk.csv")
    
    print(f"Anemia dataset shape: {anemia_df.shape}")
    print(f"Neutropenia dataset shape: {neutropenia_df.shape}")
    print(f"Thrombocytopenia dataset shape: {thrombocytopenia_df.shape}")
    
    # Prepare Anemia dataset
    print("\nPreparing Anemia dataset...")
    anemia_features = ['RBC', 'WBC', 'PLATELETS']
    anemia_X = anemia_df[anemia_features].copy()
    anemia_y = anemia_df['anemia_risk'].copy()
    
    # Handle unit conversion for anemia (PLATELETS might need conversion)
    # Check if PLATELETS are in thousands or cells/uL
    if anemia_X['PLATELETS'].mean() > 1000:
        print("  Converting PLATELETS from cells/uL to thousands/uL")
        anemia_X['PLATELETS'] = anemia_X['PLATELETS'] / 1000
    
    # Handle WBC unit conversion for anemia
    if anemia_X['WBC'].mean() > 1000:
        print("  Converting WBC from cells/uL to thousands/uL")
        anemia_X['WBC'] = anemia_X['WBC'] / 1000
    
    print(f"  Features: {anemia_features}")
    print(f"  Feature ranges - RBC: [{anemia_X['RBC'].min():.2f}, {anemia_X['RBC'].max():.2f}], "
          f"WBC: [{anemia_X['WBC'].min():.2f}, {anemia_X['WBC'].max():.2f}], "
          f"PLATELETS: [{anemia_X['PLATELETS'].min():.2f}, {anemia_X['PLATELETS'].max():.2f}]")
    print(f"  Target distribution:\n{anemia_y.value_counts()}")
    
    # Prepare Neutropenia dataset
    print("\nPreparing Neutropenia dataset...")
    neutropenia_features = ['RBC', 'WBC', 'PLATELETS', 'ANC']
    neutropenia_X = neutropenia_df[neutropenia_features].copy()
    neutropenia_y = neutropenia_df['neutropenia_risk'].copy()
    
    # Handle unit conversion for neutropenia
    # WBC and PLATELETS might be in cells/uL, convert to thousands for consistency
    if neutropenia_X['WBC'].mean() > 1000:
        print("  Converting WBC from cells/uL to thousands/uL")
        neutropenia_X['WBC'] = neutropenia_X['WBC'] / 1000
    
    if neutropenia_X['PLATELETS'].mean() > 1000:
        print("  Converting PLATELETS from cells/uL to thousands/uL")
        neutropenia_X['PLATELETS'] = neutropenia_X['PLATELETS'] / 1000
    
    # ANC is already in cells/uL, convert to thousands for consistency
    if neutropenia_X['ANC'].mean() > 1000:
        print("  Converting ANC from cells/uL to thousands/uL")
        neutropenia_X['ANC'] = neutropenia_X['ANC'] / 1000
    
    print(f"  Features: {neutropenia_features}")
    print(f"  Feature ranges - RBC: [{neutropenia_X['RBC'].min():.2f}, {neutropenia_X['RBC'].max():.2f}], "
          f"WBC: [{neutropenia_X['WBC'].min():.2f}, {neutropenia_X['WBC'].max():.2f}], "
          f"PLATELETS: [{neutropenia_X['PLATELETS'].min():.2f}, {neutropenia_X['PLATELETS'].max():.2f}], "
          f"ANC: [{neutropenia_X['ANC'].min():.2f}, {neutropenia_X['ANC'].max():.2f}]")
    print(f"  Target distribution:\n{neutropenia_y.value_counts()}")
    
    # Prepare Thrombocytopenia dataset
    print("\nPreparing Thrombocytopenia dataset...")
    thrombocytopenia_features = ['RBC', 'WBC', 'PLATELETS']
    thrombocytopenia_X = thrombocytopenia_df[thrombocytopenia_features].copy()
    thrombocytopenia_y = thrombocytopenia_df['thrombocytopenia_risk'].copy()
    
    # Handle unit conversion for thrombocytopenia
    if thrombocytopenia_X['WBC'].mean() > 1000:
        print("  Converting WBC from cells/uL to thousands/uL")
        thrombocytopenia_X['WBC'] = thrombocytopenia_X['WBC'] / 1000
    
    if thrombocytopenia_X['PLATELETS'].mean() > 1000:
        print("  Converting PLATELETS from cells/uL to thousands/uL")
        thrombocytopenia_X['PLATELETS'] = thrombocytopenia_X['PLATELETS'] / 1000
    
    print(f"  Features: {thrombocytopenia_features}")
    print(f"  Feature ranges - RBC: [{thrombocytopenia_X['RBC'].min():.2f}, {thrombocytopenia_X['RBC'].max():.2f}], "
          f"WBC: [{thrombocytopenia_X['WBC'].min():.2f}, {thrombocytopenia_X['WBC'].max():.2f}], "
          f"PLATELETS: [{thrombocytopenia_X['PLATELETS'].min():.2f}, {thrombocytopenia_X['PLATELETS'].max():.2f}]")
    print(f"  Target distribution:\n{thrombocytopenia_y.value_counts()}")
    
    return {
        'anemia': (anemia_X, anemia_y, anemia_features),
        'neutropenia': (neutropenia_X, neutropenia_y, neutropenia_features),
        'thrombocytopenia': (thrombocytopenia_X, thrombocytopenia_y, thrombocytopenia_features)
    }


def train_and_evaluate_models(X, y, feature_names, task_name):
    """Train and evaluate multiple models for a given task."""
    print("\n" + "=" * 80)
    print(f"Training models for {task_name.upper()} risk prediction")
    print("=" * 80)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            multi_class='multinomial'
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    
    # Train and evaluate models
    results = {}
    best_model = None
    best_score = 0
    best_model_name = None
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        print(f"\n{'-' * 80}")
        print(f"Training {model_name}...")
        print(f"{'-' * 80}")
        
        # Use scaled data for Logistic Regression, original for tree-based models
        if model_name == 'Logistic Regression':
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Train model
        model.fit(X_train_model, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv, scoring='accuracy')
        print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Test set evaluation
        y_pred = model.predict(X_test_model)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"  Test accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"  {cm}")
        
        results[model_name] = {
            'model': model,
            'scaler': scaler if model_name == 'Logistic Regression' else None,
            'cv_score': cv_scores.mean(),
            'test_accuracy': test_accuracy,
            'label_encoder': label_encoder
        }
        
        # Track best model
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = model
            best_model_name = model_name
            best_scaler = scaler if model_name == 'Logistic Regression' else None
    
    print(f"\n{'=' * 80}")
    print(f"Best model for {task_name}: {best_model_name} (Test Accuracy: {best_score:.4f})")
    print(f"{'=' * 80}")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_scaler': best_scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'all_results': results
    }


def save_models(results_dict):
    """Save all trained models."""
    print("\n" + "=" * 80)
    print("Saving models...")
    print("=" * 80)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for task_name, task_results in results_dict.items():
        model_file = f"models/{task_name}_risk_model_{timestamp}.pkl"
        scaler_file = f"models/{task_name}_risk_scaler_{timestamp}.pkl"
        encoder_file = f"models/{task_name}_risk_encoder_{timestamp}.pkl"
        metadata_file = f"models/{task_name}_risk_metadata_{timestamp}.txt"
        
        # Save model
        joblib.dump(task_results['best_model'], model_file)
        print(f"Saved {task_name} model: {model_file}")
        
        # Save scaler if exists
        if task_results['best_scaler'] is not None:
            joblib.dump(task_results['best_scaler'], scaler_file)
            print(f"Saved {task_name} scaler: {scaler_file}")
        
        # Save label encoder
        joblib.dump(task_results['label_encoder'], encoder_file)
        print(f"Saved {task_name} encoder: {encoder_file}")
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            f.write(f"{task_name.upper()} Risk Prediction Model\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Model Type: {task_results['best_model_name']}\n")
            f.write(f"Test Accuracy: {task_results['all_results'][task_results['best_model_name']]['test_accuracy']:.4f}\n")
            f.write(f"CV Score: {task_results['all_results'][task_results['best_model_name']]['cv_score']:.4f}\n\n")
            f.write(f"Features: {', '.join(task_results['feature_names'])}\n\n")
            f.write(f"Classes: {', '.join(task_results['label_encoder'].classes_)}\n")
        
        print(f"Saved {task_name} metadata: {metadata_file}\n")
    
    # Also save latest versions (without timestamp for easy access)
    for task_name, task_results in results_dict.items():
        model_file_latest = f"models/{task_name}_risk_model_latest.pkl"
        scaler_file_latest = f"models/{task_name}_risk_scaler_latest.pkl"
        encoder_file_latest = f"models/{task_name}_risk_encoder_latest.pkl"
        
        joblib.dump(task_results['best_model'], model_file_latest)
        if task_results['best_scaler'] is not None:
            joblib.dump(task_results['best_scaler'], scaler_file_latest)
        joblib.dump(task_results['label_encoder'], encoder_file_latest)
    
    print("Also saved 'latest' versions for easy access.")


def main():
    """Main function to train all three models."""
    print("\n" + "=" * 80)
    print("HEMATOLOGICAL RISK PREDICTION MODEL TRAINING")
    print("=" * 80)
    
    # Load and prepare data
    datasets = load_and_prepare_data()
    
    # Train models for each task
    results = {}
    
    for task_name, (X, y, features) in datasets.items():
        task_results = train_and_evaluate_models(X, y, features, task_name)
        results[task_name] = task_results
    
    # Save all models
    save_models(results)
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for task_name, task_results in results.items():
        print(f"\n{task_name.upper()}:")
        print(f"  Best Model: {task_results['best_model_name']}")
        print(f"  Test Accuracy: {task_results['all_results'][task_results['best_model_name']]['test_accuracy']:.4f}")
        print(f"  CV Score: {task_results['all_results'][task_results['best_model_name']]['cv_score']:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print("\nModels saved in the 'models' directory.")
    print("You can now use these models for prediction.")


if __name__ == "__main__":
    main()

