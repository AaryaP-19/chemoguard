# Hematological Risk Prediction Web Application

A Flask-based web application for predicting three types of hematological risks:
- **Anemia Risk**
- **Neutropenia Risk**
- **Thrombocytopenia Risk**

## Features

- 🎨 Modern, responsive web interface
- 🔬 Real-time risk predictions using trained ML models
- 📊 Probability visualization for each risk category
- 📱 Mobile-friendly design
- ⚡ Fast and accurate predictions

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure models are available:**
   Make sure the `models/` directory contains the trained model files:
   - `anemia_risk_model_latest.pkl`
   - `anemia_risk_encoder_latest.pkl`
   - `neutropenia_risk_model_latest.pkl`
   - `neutropenia_risk_encoder_latest.pkl`
   - `thrombocytopenia_risk_model_latest.pkl`
   - `thrombocytopenia_risk_encoder_latest.pkl`

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:5000`

3. **Enter patient data:**
   - RBC (Red Blood Cells) - required
   - WBC (White Blood Cells) - required
   - Platelets - required
   - ANC (Absolute Neutrophil Count) - optional (for neutropenia)
   - Neutrophils Percentage - optional (used to calculate ANC)

4. **View predictions:**
   Click "Predict Risks" to see the results with probability distributions.

## API Endpoints

### POST `/predict`
Make predictions for all three risk types.

**Request Body:**
```json
{
  "rbc": 4.5,
  "wbc": 8.0,
  "platelets": 250,
  "anc": 5.0,
  "neutrophils_percent": 60
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "anemia": {
      "risk": "Low",
      "probabilities": {
        "Low": 0.95,
        "Medium": 0.04,
        "High": 0.01
      }
    },
    "neutropenia": {
      "risk": "Low",
      "probabilities": {...}
    },
    "thrombocytopenia": {
      "risk": "Low",
      "probabilities": {...}
    }
  }
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Input Units

The application accepts values in different units and automatically converts them:

- **RBC**: millions/μL (e.g., 4.5)
- **WBC**: cells/μL (e.g., 8000) or thousands/μL (e.g., 8.0)
- **Platelets**: cells/μL (e.g., 250000) or thousands/μL (e.g., 250)
- **ANC**: cells/μL (e.g., 5000) or thousands/μL (e.g., 5.0)
- **Neutrophils**: percentage (0-100, e.g., 60)

## Project Structure

```
.
├── app.py                 # Flask application and entry point
├── auth.py                # Authentication blueprint (/api/auth)
├── routes.py              # Upload / predict / reports blueprint (/api)
├── config.py              # Configuration
├── database.py            # SQLAlchemy models
├── gpt_extraction.py      # GPT-4o-mini extraction backend
├── ocr_extraction.py      # Tesseract extraction backend
├── export.py              # PDF export and QR codes
├── init_db.py             # Database initialisation
├── templates/
│   └── index.html         # Frontend HTML
├── static/
│   ├── style.css          # Styling
│   └── script.js          # Frontend JavaScript
├── models/                # Trained ML models
│   ├── anemia_risk_model_latest.pkl
│   ├── neutropenia_risk_model_latest.pkl
│   └── thrombocytopenia_risk_model_latest.pkl
└── requirements.txt       # Python dependencies
```

## Troubleshooting

### Models not found
If you see "Model not found" error:
1. Ensure you've run `train_risk_prediction_models.py` first
2. Check that all model files exist in the `models/` directory

### Port already in use
If port 5000 is already in use, modify `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
```

### Dependencies missing
Install all required packages:
```bash
pip install Flask joblib numpy scikit-learn pandas
```

## Development

To run in development mode with auto-reload:
```bash
export FLASK_ENV=development
python app.py
```

## Production Deployment

For production, consider:
- Using a production WSGI server (e.g., Gunicorn)
- Setting `debug=False`
- Adding authentication/authorization
- Using environment variables for configuration
- Setting up proper logging

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This application is for educational and research purposes.


