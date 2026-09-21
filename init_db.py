"""
Database initialization script for ChemoGuard.
Run this script to create all database tables.
"""

from app_new import app, db
from database import User, MedicalReport, Prediction, PredictionHistory

def init_db():
    """Initialize database tables."""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
        print("Tables created:")
        print("  - users")
        print("  - medical_reports")
        print("  - predictions")
        print("  - prediction_history")

if __name__ == '__main__':
    init_db()

