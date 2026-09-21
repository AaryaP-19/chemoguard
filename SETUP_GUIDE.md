# ChemoGuard Setup Guide

This guide will help you set up the enhanced ChemoGuard application with all new features.

## Prerequisites

1. **Python 3.8+**
2. **PostgreSQL** (version 12+)
3. **Tesseract OCR** (for text extraction from images)
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL Database

1. Create a new PostgreSQL database:
```sql
CREATE DATABASE chemoguard;
```

2. Update the database URL in `.env` file (see step 3)

### 3. Configure Environment Variables

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` and update:
   - `SECRET_KEY`: Generate a random secret key
   - `DATABASE_URL`: Your PostgreSQL connection string
   - `TESSERACT_CMD`: Path to Tesseract executable (Windows only)
   - `MAIL_*`: Email configuration for sharing (optional)

### 4. Initialize Database

```bash
python init_db.py
```

This will create all necessary database tables.

### 5. Verify Models

Ensure all model files exist in the `models/` directory:
- `anemia_risk_model_latest.pkl`
- `anemia_risk_encoder_latest.pkl`
- `neutropenia_risk_model_latest.pkl`
- `neutropenia_risk_encoder_latest.pkl`
- `thrombocytopenia_risk_model_latest.pkl`
- `thrombocytopenia_risk_encoder_latest.pkl`

### 6. Create Upload Directory

```bash
mkdir uploads
```

### 7. Start the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Authentication

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/profile` - Get user profile
- `PUT /api/auth/profile` - Update user profile
- `GET /api/auth/check_auth` - Check authentication status

### Medical Reports

- `POST /api/upload` - Upload medical report (PDF/JPEG)
- `GET /api/reports` - Get all user reports

### Predictions

- `POST /api/predict` - Make prediction (with database storage)
- `GET /api/predictions` - Get prediction history
- `GET /api/predictions/<id>` - Get specific prediction

### Export & Sharing

- `GET /api/export/pdf/<prediction_id>` - Export PDF report
- `GET /api/share/qr/<prediction_id>` - Get QR code for sharing

### Legacy (Backward Compatibility)

- `POST /predict` - Legacy predict endpoint (no auth required)
- `GET /health` - Health check
- `POST /reload_models` - Reload ML models

## Testing the API

### 1. Register a User

```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "test123",
    "full_name": "Test User"
  }'
```

### 2. Login

```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "test123"
  }'
```

Save the session cookie from the response.

### 3. Upload a Report

```bash
curl -X POST http://localhost:5000/api/upload \
  -H "Cookie: session=<your-session-cookie>" \
  -F "file=@/path/to/report.pdf"
```

### 4. Make a Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -H "Cookie: session=<your-session-cookie>" \
  -d '{
    "rbc": 4.5,
    "wbc": 7.0,
    "platelets": 250,
    "anc": 3.5
  }'
```

### 5. Export PDF

```bash
curl -X GET http://localhost:5000/api/export/pdf/1 \
  -H "Cookie: session=<your-session-cookie>" \
  -o report.pdf
```

## Features Implemented

✅ **User Management**
- Registration and login
- User profiles
- Session management

✅ **Medical Report Upload**
- PDF and JPEG support
- File validation and storage

✅ **OCR Extraction**
- Automatic parameter extraction from reports
- OpenCV image preprocessing
- Tesseract OCR integration

✅ **Toxicity Prediction**
- Anemia risk prediction
- Neutropenia risk prediction
- Thrombocytopenia risk prediction
- Combined severity calculation

✅ **Database Storage**
- User data
- Medical reports
- Prediction history
- Versioned history tracking

✅ **Export & Sharing**
- PDF report generation
- QR code generation for sharing

## Next Steps

1. **Frontend Updates**: Update the web interface to support new features
2. **Email Sharing**: Implement email sending for report sharing
3. **Trend Graphs**: Add visualization for prediction trends over time
4. **Early Warning System**: Implement alerts for rising toxicity risk

## Troubleshooting

### Models Not Loading
- Ensure all model files exist in `models/` directory
- Check file permissions
- Verify model files are not corrupted

### Database Connection Error
- Verify PostgreSQL is running
- Check database URL in `.env`
- Ensure database exists and user has permissions

### OCR Not Working
- Verify Tesseract is installed
- Check `TESSERACT_CMD` path in `.env` (Windows)
- Test Tesseract: `tesseract --version`

### File Upload Errors
- Check `uploads/` directory exists and is writable
- Verify file size is under 16MB
- Ensure file type is allowed (PDF, JPEG, PNG)

## Support

For issues or questions, please check the project documentation or create an issue in the repository.

