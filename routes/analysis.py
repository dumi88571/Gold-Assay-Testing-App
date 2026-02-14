"""Analysis routes for sample prediction and processing."""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from pydantic import ValidationError

from app import db
from app.models import Sample, Batch, AuditLog
from app.schemas import SampleInput
from app.services.ml_engine import get_ml_engine

bp = Blueprint('analysis', __name__)


@bp.route('/')
@login_required
def index():
    """Analysis form page."""
    return render_template('analysis_form.html')


@bp.route('/submit', methods=['POST'])
@login_required
def submit():
    """Submit single sample for analysis."""
    try:
        # Validate input
        data = request.form.to_dict()
        sample_input = SampleInput(**data)
        
        # Create sample record
        sample = Sample()
        sample.generate_sample_id()
        sample.analyst_id = current_user.id
        sample.sample_type = sample_input.sample_type
        sample.assay_method = sample_input.assay_method
        sample.client_name = sample_input.client_name
        
        # Set all 17 features
        for feature in [
            'absorption_242nm', 'absorption_267nm', 'emission_intensity',
            'solution_ph', 'temperature_c', 'ionic_strength', 'dissolved_oxygen',
            'iron_ppm', 'copper_ppm', 'silver_ppm', 'sulfur_content',
            'dilution_factor', 'measurement_time', 'calibration_drift',
            'replicate_rsd', 'blank_intensity', 'internal_standard'
        ]:
            setattr(sample, feature, getattr(sample_input, feature))
        
        # Get prediction from ML engine
        ml_engine = get_ml_engine()
        prediction = ml_engine.predict(sample_input.dict())
        
        sample.predicted_grade = prediction['grade']
        # New model returns R² as confidence instead of old confidence %
        sample.model_confidence = ml_engine.metrics.get('r2_score', 0.95) * 100
        sample.uncertainty = prediction['uncertainty']
        sample.qc_passed = len(prediction['qc_flags']) == 0
        sample.qc_flags = prediction['qc_flags']
        sample.status = 'completed'
        
        if sample_input.actual_grade:
            sample.actual_grade = sample_input.actual_grade
        
        db.session.add(sample)
        db.session.commit()
        
        # Log action
        AuditLog.log_action(
            action='create',
            entity_type='Sample',
            entity_id=sample.sample_id,
            new_values=sample.to_dict(),
            notes='Single sample analysis'
        )
        
        flash(f'Analysis complete. Sample ID: {sample.sample_id}', 'success')
        return redirect(url_for('main.sample_detail', sample_id=sample.sample_id))
        
    except ValidationError as e:
        flash(f'Validation error: {e.errors()}', 'danger')
        return redirect(url_for('analysis.index'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error processing sample: {str(e)}', 'danger')
        return redirect(url_for('analysis.index'))


@bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for prediction without saving."""
    try:
        data = request.get_json()
        sample_input = SampleInput(**data)
        
        ml_engine = get_ml_engine()
        prediction = ml_engine.predict(sample_input.dict())
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
        
    except ValidationError as e:
        return jsonify({
            'status': 'error',
            'message': 'Validation error',
            'errors': e.errors()
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@bp.route('/api/predict-save', methods=['POST'])
@login_required
def api_predict_save():
    """API endpoint for prediction and saving."""
    try:
        data = request.get_json()
        sample_input = SampleInput(**data)
        
        # Create sample
        sample = Sample()
        sample.generate_sample_id()
        sample.analyst_id = current_user.id
        sample.sample_type = sample_input.sample_type
        sample.assay_method = sample_input.assay_method
        sample.client_name = sample_input.client_name
        
        # Set features
        for feature in [
            'absorption_242nm', 'absorption_267nm', 'emission_intensity',
            'solution_ph', 'temperature_c', 'ionic_strength', 'dissolved_oxygen',
            'iron_ppm', 'copper_ppm', 'silver_ppm', 'sulfur_content',
            'dilution_factor', 'measurement_time', 'calibration_drift',
            'replicate_rsd', 'blank_intensity', 'internal_standard'
        ]:
            setattr(sample, feature, getattr(sample_input, feature))
        
        # Predict
        ml_engine = get_ml_engine()
        prediction = ml_engine.predict(sample_input.dict())
        
        sample.predicted_grade = prediction['grade']
        # New model returns R² as confidence instead of old confidence %
        sample.model_confidence = ml_engine.metrics.get('r2_score', 0.95) * 100
        sample.uncertainty = prediction['uncertainty']
        sample.qc_passed = len(prediction['qc_flags']) == 0
        sample.qc_flags = prediction['qc_flags']
        sample.status = 'completed'
        
        if sample_input.actual_grade:
            sample.actual_grade = sample_input.actual_grade
        
        db.session.add(sample)
        db.session.commit()
        
        AuditLog.log_action(
            action='create',
            entity_type='Sample',
            entity_id=sample.sample_id,
            new_values=sample.to_dict()
        )
        
        return jsonify({
            'status': 'success',
            'sample_id': sample.sample_id,
            'prediction': prediction
        })
        
    except ValidationError as e:
        return jsonify({
            'status': 'error',
            'message': 'Validation error',
            'errors': e.errors()
        }), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@bp.route('/sample/<int:id>/update', methods=['POST'])
@login_required
def update_sample(id):
    """Update sample with actual grade (for training)."""
    sample = Sample.query.get_or_404(id)
    
    try:
        old_values = sample.to_dict()
        
        actual_grade = request.form.get('actual_grade', type=float)
        if actual_grade is not None:
            sample.actual_grade = actual_grade
            sample.status = 'validated'
            
            db.session.commit()
            
            AuditLog.log_action(
                action='update',
                entity_type='Sample',
                entity_id=sample.sample_id,
                old_values=old_values,
                new_values=sample.to_dict(),
                notes='Updated with actual grade for validation'
            )
            
            flash('Sample updated with actual grade', 'success')
        
        return redirect(url_for('main.sample_detail', sample_id=sample.sample_id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating sample: {str(e)}', 'danger')
        return redirect(url_for('main.sample_detail', sample_id=sample.sample_id))
