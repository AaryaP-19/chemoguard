"""
Main routes for ChemoGuard application including file upload, prediction, and reporting.
"""

from flask import Blueprint, request, jsonify, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from cbc import panel_from_mapping
from database import (
    db,
    MedicalReport,
    Prediction,
    PredictionHistory,
    PREDICTION_INPUT_ANALYTES,
    REPORT_ANALYTES,
    model_feature_units,
    panel_from_columns,
    prediction_input_columns,
    report_columns,
    reported_quantities,
)
from gpt_extraction import GPTExtractor
from clinical_flags import flags_for
from datetime import datetime
import os
import json
import prediction_log

routes_bp = Blueprint('routes', __name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@routes_bp.route('/upload', methods=['POST'])
@login_required
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
        user_id = current_user.id
        unique_filename = f"{user_id}_{timestamp}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(file_path)
        
        # Get file type
        file_type = filename.rsplit('.', 1)[1].lower()
        
        # Extract parameters using GPT-4o-mini. The extractor returns a
        # CBCPanel: every value it found carries the unit string printed on
        # the report. A value whose unit could not be read is not returned
        # at all, rather than returned as a bare number.
        try:
            extractor = GPTExtractor()
            panel, confidence, skipped = extractor.process_file(file_path, file_type)
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
        
        # Persist the report for the authenticated user.
        report_id = None
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

            # Canonical values into the unit-named columns, and the report's
            # own wording into reported_quantities alongside them.
            for column, value in report_columns(panel).items():
                setattr(report, column, value)
            report.reported_quantities = reported_quantities(panel)
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
            'extracted_parameters': panel.as_dict(),
            # Values the extractor found but could not attach a unit to.
            # Reported rather than silently dropped: the operator needs to
            # know the report was read and the unit was not, so they can
            # supply it deliberately.
            'unusable_without_units': skipped,
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
                # Values leave with their units attached, same as they
                # arrived. A consumer of this endpoint never has to know a
                # convention to read a number.
                'values': panel_from_columns(r, REPORT_ANALYTES).as_dict(),
                'as_reported': r.reported_quantities,
                'ocr_extracted': r.ocr_extracted,
                'ocr_confidence': r.ocr_confidence
            } for r in reports]
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint with database storage (works with or without auth).

    Every CBC value must arrive with a unit -- see cbc.panel_from_mapping for
    the three accepted shapes. A bare number returns 422 and is logged as a
    refusal; it is never interpreted.
    """
    request_id = prediction_log.new_prediction_id()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    report_id = data.get('report_id')  # Optional: link to uploaded report
    user_id = current_user.id if current_user.is_authenticated else None

    # Deferred import: the application module imports this blueprint at
    # load time, so importing it at module scope would be circular.
    from app import (
        ModelsUnavailable,
        combine_severity,
        predict_risk,
        prediction_error_response,
    )

    try:
        panel = panel_from_mapping(data)

        missing = panel.missing_required()
        if missing:
            return jsonify({
                'error': 'RBC, WBC, and Platelets are required',
                'missing': missing,
                'request_id': request_id,
            }), 400

        predictions, flags, _ = predict_risk(
            panel, request_id=request_id, user_id=user_id, report_id=report_id
        )
    except ModelsUnavailable as exc:
        # The CTCAE grading is a rules engine and needs no model, so a
        # missing artefact should not cost the caller the grading too.
        # Nothing is stored: a grading-only result is not a prediction.
        prediction_log.log_refusal(
            request_id=request_id,
            reason='models_unavailable',
            detail=str(exc),
            raw_input=data,
            user_id=user_id,
        )
        return jsonify({
            'success': True,
            'request_id': request_id,
            'prediction_id': None,
            'predictions': None,
            'models_available': False,
            'notice': ('Statistical risk prediction is unavailable: the '
                       'previous models were withdrawn. Clinical grading '
                       'below is rule-based (CTCAE v5.0) and unaffected.'),
            'flags': [f.as_dict() for f in flags_for(panel)],
            'input_values': panel.as_dict(),
        }), 200
    except Exception as exc:
        handled = prediction_error_response(exc, request_id, data, user_id=user_id)
        if handled is not None:
            return handled
        db.session.rollback()
        return jsonify({'error': str(exc), 'request_id': request_id}), 500

    combined = combine_severity(predictions)
    combined_severity = combined['severity']
    max_score = combined['score']

    def probability(model, level):
        body = predictions.get(model) or {}
        return (body.get('probabilities') or {}).get(level)

    # Store prediction in database only if user is authenticated
    prediction_id = None
    if current_user.is_authenticated:
        try:
            prediction = Prediction(
                user_id=current_user.id,
                report_id=report_id if report_id else None,
                request_id=request_id,
                # Canonical values, in columns named for the unit they hold,
                # plus the as-recorded originals and the units each estimator
                # was actually fed.
                **prediction_input_columns(panel),
                input_reported_quantities=reported_quantities(panel),
                model_feature_units=model_feature_units(predictions),
                anemia_risk=predictions['anemia']['risk'],
                anemia_probability_low=probability('anemia', 'Low'),
                anemia_probability_medium=probability('anemia', 'Medium'),
                anemia_probability_high=probability('anemia', 'High'),
                neutropenia_risk=predictions.get('neutropenia', {}).get('risk'),
                neutropenia_probability_low=probability('neutropenia', 'Low'),
                neutropenia_probability_medium=probability('neutropenia', 'Medium'),
                neutropenia_probability_high=probability('neutropenia', 'High'),
                thrombocytopenia_risk=predictions['thrombocytopenia']['risk'],
                thrombocytopenia_probability_low=probability('thrombocytopenia', 'Low'),
                thrombocytopenia_probability_medium=probability('thrombocytopenia', 'Medium'),
                thrombocytopenia_probability_high=probability('thrombocytopenia', 'High'),
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
                    'input_values': panel.as_dict(),
                    'flags': [f.as_dict() for f in flags],
                    'request_id': request_id,
                })
            )
            db.session.add(history)
            db.session.commit()
        except Exception as db_error:
            db.session.rollback()
            print(f"Warning: Could not save prediction to database: {db_error}")

    return jsonify({
        'success': True,
        'request_id': request_id,
        'prediction_id': prediction_id,
        'predictions': predictions,
        'combined_severity': combined_severity,
        'combined_toxicity': combined_severity,  # Alias for clarity
        'combined': combined,
        'flags': [f.as_dict() for f in flags],
        'input_values': panel.as_dict(),
    }), 200


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
                'input_values': panel_from_columns(
                    p, PREDICTION_INPUT_ANALYTES, prefix='input_'
                ).as_dict(),
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
                'input_values': panel_from_columns(
                    prediction, PREDICTION_INPUT_ANALYTES, prefix='input_'
                ).as_dict(),
                'input_as_reported': prediction.input_reported_quantities,
                'model_feature_units': prediction.model_feature_units,
                'request_id': prediction.request_id,
                'report_id': prediction.report_id
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500