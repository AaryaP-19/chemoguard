> ## WITHDRAWN — do not rely on anything below
>
> **These models have been quarantined and are no longer in the repository.**
> They were moved on 2026-09-21 to
> `C:\Users\tanii\Downloads\chemoguard-withdrawn-dataset\models\`, retained
> only so the audit trail is reproducible. Nothing should train on them,
> load them, or cite their numbers.
>
> **The dataset was withdrawn.** The 7,196 samples referenced throughout this
> file came from a Mendeley Data record (DOI `10.17632/rnfzzy4wz6`) that has
> since been removed at the author's request. No licence, collection method,
> institution or validation procedure survives it. Provenance could not be
> established.
>
> **The accuracy figures below are artefacts, not performance.** Every label
> was a threshold rule computed from the same columns handed to the model as
> features — `anemia_risk` from RBC/WBC/Platelets, `neutropenia_risk` from
> ANC, `thrombocytopenia_risk` from Platelets. A depth-3 decision tree
> reproduces every published label on 100.00% of rows. The ">99% accuracy"
> claim measures arithmetic, not learning, and the majority-class baselines
> were already 69%, 92% and 91%.
>
> The data does not hold together either: five CBC identity checks fail, and
> roughly 90% of rows fail them jointly.
>
> See `CLAUDE.md`, Findings 1, 5, 6 and 7. Replacements will be trained on
> NHANES.

# Hematological Risk Prediction Models

This directory contains trained machine learning models for predicting three types of hematological risks:

1. **Anemia Risk** - Predicts Low, Medium, or High risk
2. **Neutropenia Risk** - Predicts Low, Medium, or High risk  
3. **Thrombocytopenia Risk** - Predicts Low, Medium, or High risk

## Model Performance

### Anemia Risk Model
- **Best Model**: Random Forest
- **Test Accuracy**: 99.93%
- **Cross-Validation Score**: 99.95%
- **Features**: RBC, WBC, Platelets

### Neutropenia Risk Model
- **Best Model**: Gradient Boosting
- **Test Accuracy**: 99.93%
- **Cross-Validation Score**: 99.98%
- **Features**: RBC, WBC, Platelets, ANC

### Thrombocytopenia Risk Model
- **Best Model**: Gradient Boosting
- **Test Accuracy**: 100.00%
- **Cross-Validation Score**: 100.00%
- **Features**: RBC, WBC, Platelets

## Model Files

Each model has three associated files:
- `{risk_type}_risk_model_latest.pkl` - The trained model
- `{risk_type}_risk_encoder_latest.pkl` - Label encoder for risk categories
- `{risk_type}_risk_scaler_latest.pkl` - Feature scaler (if applicable)

## Usage

### Using the Prediction Script

```bash
# Basic usage with standard units
python predict_risk.py --rbc 4.5 --wbc 8.0 --platelets 250 --anc 5.0

# Using values in cells/μL (will be converted automatically)
python predict_risk.py --rbc 4.5 --wbc 8000 --platelets 250000 --anc 5000

# Calculate ANC from WBC and neutrophil percentage
python predict_risk.py --rbc 4.5 --wbc 8000 --platelets 250000 --neutrophils 60
```

### Using Models Programmatically

```python
import joblib
import numpy as np

# Load models
anemia_model = joblib.load('models/anemia_risk_model_latest.pkl')
anemia_encoder = joblib.load('models/anemia_risk_encoder_latest.pkl')

# Prepare features (RBC, WBC, Platelets in thousands/μL)
features = np.array([[4.5, 8.0, 250]])  # Example values

# Make prediction
prediction = anemia_model.predict(features)[0]
risk_category = anemia_encoder.inverse_transform([prediction])[0]

print(f"Predicted Anemia Risk: {risk_category}")
```

## Feature Units

The models expect features in the following units:
- **RBC**: millions/μL (e.g., 4.5)
- **WBC**: thousands/μL (e.g., 8.0 for 8000 cells/μL)
- **Platelets**: thousands/μL (e.g., 250 for 250,000 cells/μL)
- **ANC**: thousands/μL (e.g., 5.0 for 5000 cells/μL)

The prediction script automatically handles unit conversion if values are provided in cells/μL.

## Retraining Models

To retrain the models with updated data:

```bash
python train_risk_prediction_models.py
```

This will:
1. Load the three datasets (anemia, neutropenia, thrombocytopenia)
2. Train multiple models (Random Forest, Gradient Boosting, Logistic Regression, XGBoost)
3. Select the best model for each task
4. Save the models with timestamps and as "latest" versions

## Model Details

### Anemia Risk Model
- Uses RBC, WBC, and Platelets to predict anemia risk
- Thresholds used in original rule-based system:
  - RBC < 3.5 → flag
  - WBC < 4.0 → flag
  - Platelets < 150 → flag
  - Risk based on sum of flags (0=Low, 1=Medium, ≥2=High)

### Neutropenia Risk Model
- Uses RBC, WBC, Platelets, and ANC to predict neutropenia risk
- ANC thresholds:
  - ANC ≥ 1500: Low risk
  - 1000 ≤ ANC < 1500: Medium risk
  - 500 ≤ ANC < 1000: High risk
  - ANC < 500: High risk

### Thrombocytopenia Risk Model
- Uses RBC, WBC, and Platelets to predict thrombocytopenia risk
- Platelet thresholds:
  - Platelets ≥ 150: Low risk
  - 75 ≤ Platelets < 150: Medium risk
  - Platelets < 75: High risk

## Requirements

- pandas
- numpy
- scikit-learn
- joblib
- xgboost (optional, but recommended)

Install with:
```bash
pip install pandas numpy scikit-learn joblib xgboost
```

## Notes

- Models were trained on 7,196 samples from a **withdrawn** dataset (see the
  notice at the top of this file)
- 80/20 train/test split was used
- 5-fold cross-validation was performed
- Class weights were balanced to handle imbalanced datasets
- ~~All models achieved excellent performance (>99% accuracy)~~ — this is
  false. The labels were threshold rules over the models' own input
  features, so >99% was arithmetic. Report accuracy against the
  majority-class baseline (69% / 92% / 91%), never on its own.


