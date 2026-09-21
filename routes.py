"""
Main routes for ChemoGuard application including file upload, prediction, and reporting.
"""

from flask import Blueprint, request, jsonify, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from database import db, MedicalReport, Prediction, PredictionHistory
from gpt_extraction import GPTExtractor
from datetime import datetime
import os
import json

routes_bp = Blueprint('routes', __name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@routes_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload medical report file and extract parameters using GPT-4o-mini."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types: {", ".join(current_app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_id = current_user.id if current_user.is_authenticated else 0
        unique_filename = f"{user_id}_{timestamp}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(file_path)
        
        # Get file type
        file_type = filename.rsplit('.', 1)[1].lower()
        
        # Extract parameters using GPT-4o-mini
        try:
            extractor = GPTExtractor()
            parameters, confidence = extractor.process_file(file_path, file_type)
        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            return jsonify({
                'error': f'Failed to extract parameters: {str(e)}'
            }), 500
        
        # Only save to database if user is authenticated
        report_id = None
        if current_user.is_authenticated:
            try:
                report = MedicalReport(
                    user_id=current_user.id,
                    filename=filename,
                    file_path=file_path,
                    file_type=file_type
                )
                db.session.add(report)
                db.session.flush()
                report_id = report.id
                
                # Update report with extracted values
                report.hb = parameters.get('hb')
                report.rbc = parameters.get('rbc')
                report.wbc = parameters.get('wbc')
                report.anc = parameters.get('anc')
                report.platelets = parameters.get('platelets')
                report.neutrophils_percent = parameters.get('neutrophils_percent')
                report.hematocrit = parameters.get('hematocrit')
                report.mcv = parameters.get('mcv')
                report.mch = parameters.get('mch')
                report.mchc = parameters.get('mchc')
                report.ocr_extracted = True
                report.ocr_confidence = confidence
                report.extraction_date = datetime.utcnow()
                report.raw_ocr_text = f"GPT-4o-mini extraction confidence: {confidence:.1f}%"
                
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                # Continue even if DB save fails
                print(f"Warning: Could not save to database: {db_error}")
        
        return jsonify({
            'success': True,
            'message': 'File processed successfully',
            'report_id': report_id,
            'extracted_parameters': {k: v for k, v in parameters.items() if v is not None},
            'confidence': confidence
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/reports', methods=['GET'])
@login_required
def get_reports():
    """Get all reports for current user."""
    try:
        reports = MedicalReport.query.filter_by(user_id=current_user.id)\
            .order_by(MedicalReport.upload_date.desc()).all()
        
        return jsonify({
            'success': True,
            'reports': [{
                'id': r.id,
                'filename': r.filename,
                'file_type': r.file_type,
                'upload_date': r.upload_date.isoformat() if r.upload_date else None,
                'rbc': r.rbc,
                'wbc': r.wbc,
                'platelets': r.platelets,
                'anc': r.anc,
                'hb': r.hb,
                'ocr_extracted': r.ocr_extracted,
                'ocr_confidence': r.ocr_confidence
            } for r in reports]
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/predict', methods=['POST'])
def predict():
    """Enhanced predict endpoint with database storage (works with or without auth)."""
    try:
        data = request.get_json()
        
        # Extract input values
        rbc = float(data.get('rbc', 0))
        wbc = float(data.get('wbc', 0))
        platelets = float(data.get('platelets', 0))
        anc = data.get('anc')
        anc = float(anc) if anc else None
        neutrophils_percent = data.get('neutrophils_percent')
        neutrophils_percent = float(neutrophils_percent) if neutrophils_percent else None
        report_id = data.get('report_id')  # Optional: link to uploaded report
        
        # Validate required inputs
        if not all([rbc, wbc, platelets]):
            return jsonify({
                'error': 'RBC, WBC, and Platelets are required'
            }), 400
        
        # Deferred import: the application module imports this blueprint at
        # load time, so importing it at module scope would be circular.
        from app import predict_risk
        
        predictions = predict_risk(rbc, wbc, platelets, anc, neutrophils_percent)
        
        # Calculate combined severity/toxicity
        risk_scores = {'Low': 1, 'Medium': 2, 'High': 3}
        anemia_score = risk_scores.get(predictions['anemia']['risk'], 1)
        neutropenia_score = risk_scores.get(predictions.get('neutropenia', {}).get('risk', 'Low'), 1)
        thrombocytopenia_score = risk_scores.get(predictions['thrombocytopenia']['risk'], 1)
        
        max_score = max(anemia_score, neutropenia_score, thrombocytopenia_score)
        combined_severity = {1: 'Low', 2: 'Moderate', 3: 'High'}[max_score]
        
        # Store prediction in database only if user is authenticated
        prediction_id = None
        if current_user.is_authenticated:
            try:
                prediction = Prediction(
                    user_id=current_user.id,
                    report_id=report_id if report_id else None,
                    input_rbc=rbc,
                    input_wbc=wbc,
                    input_platelets=platelets,
                    input_anc=anc,
                    input_neutrophils_percent=neutrophils_percent,
                    anemia_risk=predictions['anemia']['risk'],
                    anemia_probability_low=predictions['anemia']['probabilities'].get('Low') if predictions['anemia']['probabilities'] else None,
                    anemia_probability_medium=predictions['anemia']['probabilities'].get('Medium') if predictions['anemia']['probabilities'] else None,
                    anemia_probability_high=predictions['anemia']['probabilities'].get('High') if predictions['anemia']['probabilities'] else None,
                    neutropenia_risk=predictions.get('neutropenia', {}).get('risk', 'N/A'),
                    neutropenia_probability_low=predictions.get('neutropenia', {}).get('probabilities', {}).get('Low') if predictions.get('neutropenia', {}).get('probabilities') else None,
                    neutropenia_probability_medium=predictions.get('neutropenia', {}).get('probabilities', {}).get('Medium') if predictions.get('neutropenia', {}).get('probabilities') else None,
                    neutropenia_probability_high=predictions.get('neutropenia', {}).get('probabilities', {}).get('High') if predictions.get('neutropenia', {}).get('probabilities') else None,
                    thrombocytopenia_risk=predictions['thrombocytopenia']['risk'],
                    thrombocytopenia_probability_low=predictions['thrombocytopenia']['probabilities'].get('Low') if predictions['thrombocytopenia']['probabilities'] else None,
                    thrombocytopenia_probability_medium=predictions['thrombocytopenia']['probabilities'].get('Medium') if predictions['thrombocytopenia']['probabilities'] else None,
                    thrombocytopenia_probability_high=predictions['thrombocytopenia']['probabilities'].get('High') if predictions['thrombocytopenia']['probabilities'] else None,
                    combined_severity=combined_severity,
                    combined_score=max_score
                )
                
                db.session.add(prediction)
                db.session.flush()
                prediction_id = prediction.id
                
                # Create history entry
                history = PredictionHistory(
                    prediction_id=prediction.id,
                    user_id=current_user.id,
                    version=1,
                    prediction_data=json.dumps({
                        'predictions': predictions,
                        'input_values': {
                            'rbc': rbc,
                            'wbc': wbc,
                            'platelets': platelets,
                            'anc': anc,
                            'neutrophils_percent': neutrophils_percent
                        }
                    })
                )
                db.session.add(history)
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                print(f"Warning: Could not save prediction to database: {db_error}")
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
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
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'error': str(e)
        }), 500


@routes_bp.route('/predictions', methods=['GET'])
@login_required
def get_predictions():
    """Get prediction history for current user."""
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id)\
            .order_by(Prediction.prediction_date.desc()).all()
        
        return jsonify({
            'success': True,
            'predictions': [{
                'id': p.id,
                'prediction_date': p.prediction_date.isoformat() if p.prediction_date else None,
                'anemia_risk': p.anemia_risk,
                'neutropenia_risk': p.neutropenia_risk,
                'thrombocytopenia_risk': p.thrombocytopenia_risk,
                'combined_severity': p.combined_severity,
                'input_rbc': p.input_rbc,
                'input_wbc': p.input_wbc,
                'input_platelets': p.input_platelets,
                'report_id': p.report_id
            } for p in predictions]
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/predictions/<int:prediction_id>', methods=['GET'])
@login_required
def get_prediction(prediction_id):
    """Get detailed prediction by ID."""
    try:
        prediction = Prediction.query.filter_by(
            id=prediction_id,
            user_id=current_user.id
        ).first_or_404()
        
        return jsonify({
            'success': True,
            'prediction': {
                'id': prediction.id,
                'prediction_date': prediction.prediction_date.isoformat() if prediction.prediction_date else None,
                'anemia': {
                    'risk': prediction.anemia_risk,
                    'probabilities': {
                        'Low': prediction.anemia_probability_low,
                        'Medium': prediction.anemia_probability_medium,
                        'High': prediction.anemia_probability_high
                    }
                },
                'neutropenia': {
                    'risk': prediction.neutropenia_risk,
                    'probabilities': {
                        'Low': prediction.neutropenia_probability_low,
                        'Medium': prediction.neutropenia_probability_medium,
                        'High': prediction.neutropenia_probability_high
                    }
                },
                'thrombocytopenia': {
                    'risk': prediction.thrombocytopenia_risk,
                    'probabilities': {
                        'Low': prediction.thrombocytopenia_probability_low,
                        'Medium': prediction.thrombocytopenia_probability_medium,
                        'High': prediction.thrombocytopenia_probability_high
                    }
                },
                'combined_severity': prediction.combined_severity,
                'combined_score': prediction.combined_score,
                'input_values': {
                    'rbc': prediction.input_rbc,
                    'wbc': prediction.input_wbc,
                    'platelets': prediction.input_platelets,
                    'anc': prediction.input_anc,
                    'neutrophils_percent': prediction.input_neutrophils_percent
                },
                'report_id': prediction.report_id
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

