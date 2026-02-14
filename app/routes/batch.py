"""Batch processing routes for multiple samples."""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
import json

from app import db
from app.models import Sample, Batch, AuditLog
from app.schemas import BatchInput, SampleInput
from app.services.ml_engine import get_ml_engine
from app.utils.helpers import (
    validate_file_extension, compute_file_hash,
    parse_csv_batch, parse_excel_batch, json_response, error_response
)

bp = Blueprint('batch', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}


@bp.route('/')
@login_required
def index():
    """Batch processing dashboard."""
    # Recent batches
    batches = Batch.query.order_by(Batch.created_at.desc()).limit(10).all()
    return render_template('batch_index.html', batches=batches)


@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Upload file for batch processing."""
    if request.method == 'GET':
        return render_template('batch_upload.html')
    
    # Handle file upload
    if 'file' not in request.files:
        flash('No file provided', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(request.url)
    
    if not validate_file_extension(file.filename, ALLOWED_EXTENSIONS):
        flash('Invalid file type. Use CSV, Excel, or JSON', 'danger')
        return redirect(request.url)
    
    try:
        # Parse file based on type
        ext = file.filename.rsplit('.', 1)[1].lower()
        
        if ext == 'csv':
            samples_data = parse_csv_batch(file)
        elif ext == 'xlsx':
            samples_data = parse_excel_batch(file)
        elif ext == 'json':
            samples_data = json.loads(file.read().decode('utf-8'))
            if not isinstance(samples_data, list):
                samples_data = [samples_data]
        else:
            flash('Unsupported file format', 'danger')
            return redirect(request.url)
        
        # Create batch
        batch = Batch()
        batch.generate_batch_id()
        batch.batch_name = request.form.get('batch_name', '')
        batch.client_name = request.form.get('client_name', '')
        batch.sample_count = len(samples_data)
        batch.source_filename = secure_filename(file.filename)
        batch.status = 'pending'
        
        db.session.add(batch)
        db.session.commit()
        
        # Store samples data for processing
        batch._samples_data = samples_data
        
        flash(f'Batch {batch.batch_id} created with {len(samples_data)} samples', 'success')
        return redirect(url_for('batch.process', batch_id=batch.batch_id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error processing file: {str(e)}', 'danger')
        return redirect(request.url)


@bp.route('/process/<batch_id>', methods=['GET', 'POST'])
@login_required
def process(batch_id):
    """Process batch samples."""
    batch = Batch.query.filter_by(batch_id=batch_id).first_or_404()
    
    if request.method == 'POST':
        # Get samples data from hidden field or re-parse
        samples_data = request.form.get('samples_data')
        if samples_data:
            samples_data = json.loads(samples_data)
        else:
            flash('No sample data found', 'danger')
            return redirect(url_for('batch.index'))
        
        # Process each sample
        ml_engine = get_ml_engine()
        completed = 0
        failed = 0
        grades = []
        
        batch.status = 'processing'
        db.session.commit()
        
        for sample_data in samples_data:
            try:
                # Validate and create sample
                sample_input = SampleInput(**sample_data)
                
                sample = Sample()
                sample.generate_sample_id()
                sample.batch_id = batch_id
                sample.analyst_id = current_user.id
                sample.sample_type = sample_input.sample_type
                sample.assay_method = sample_input.assay_method
                
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
                prediction = ml_engine.predict(sample_input.dict())
                sample.predicted_grade = prediction['grade']
                sample.model_confidence = prediction['confidence']
                sample.uncertainty = prediction['uncertainty']
                sample.qc_passed = len(prediction['qc_flags']) == 0
                sample.qc_flags = prediction['qc_flags']
                sample.status = 'completed'
                
                db.session.add(sample)
                grades.append(prediction['grade'])
                completed += 1
                
            except Exception as e:
                failed += 1
                current_app.logger.error(f'Batch processing error: {e}')
                continue
        
        # Update batch summary
        batch.status = 'completed'
        batch.completed_at = datetime.utcnow()
        batch.completed_count = completed
        batch.failed_count = failed
        
        if grades:
            batch.avg_grade = sum(grades) / len(grades)
            batch.min_grade = min(grades)
            batch.max_grade = max(grades)
        
        db.session.commit()
        
        AuditLog.log_action(
            action='batch_process',
            entity_type='Batch',
            entity_id=batch_id,
            notes=f'Processed {completed} samples, {failed} failed'
        )
        
        flash(f'Batch processing complete. {completed} successful, {failed} failed', 'success')
        return redirect(url_for('batch.results', batch_id=batch_id))
    
    return render_template('batch_process.html', batch=batch)


@bp.route('/results/<batch_id>')
@login_required
def results(batch_id):
    """View batch processing results."""
    batch = Batch.query.filter_by(batch_id=batch_id).first_or_404()
    samples = Sample.query.filter_by(batch_id=batch_id).all()
    
    return render_template('batch_results.html', batch=batch, samples=samples)


@bp.route('/api/submit', methods=['POST'])
@login_required
def api_submit_batch():
    """API endpoint for batch submission."""
    try:
        data = request.get_json()
        batch_input = BatchInput(**data)
        
        # Create batch
        batch = Batch()
        batch.generate_batch_id()
        batch.batch_name = batch_input.batch_name
        batch.client_name = batch_input.client_name
        batch.sample_count = len(batch_input.samples)
        batch.status = 'processing'
        
        db.session.add(batch)
        
        # Process samples
        ml_engine = get_ml_engine()
        grades = []
        sample_results = []
        
        for sample_input in batch_input.samples:
            sample = Sample()
            sample.generate_sample_id()
            sample.batch_id = batch.batch_id
            sample.analyst_id = current_user.id
            sample.sample_type = sample_input.sample_type
            sample.assay_method = sample_input.assay_method
            
            for feature in [
                'absorption_242nm', 'absorption_267nm', 'emission_intensity',
                'solution_ph', 'temperature_c', 'ionic_strength', 'dissolved_oxygen',
                'iron_ppm', 'copper_ppm', 'silver_ppm', 'sulfur_content',
                'dilution_factor', 'measurement_time', 'calibration_drift',
                'replicate_rsd', 'blank_intensity', 'internal_standard'
            ]:
                setattr(sample, feature, getattr(sample_input, feature))
            
            prediction = ml_engine.predict(sample_input.dict())
            sample.predicted_grade = prediction['grade']
            sample.model_confidence = prediction['confidence']
            sample.uncertainty = prediction['uncertainty']
            sample.qc_passed = len(prediction['qc_flags']) == 0
            sample.qc_flags = prediction['qc_flags']
            sample.status = 'completed'
            
            db.session.add(sample)
            grades.append(prediction['grade'])
            sample_results.append({
                'sample_id': sample.sample_id,
                'grade': prediction['grade'],
                'qc_flags': prediction['qc_flags']
            })
        
        batch.status = 'completed'
        batch.completed_at = datetime.utcnow()
        batch.completed_count = len(sample_results)
        
        if grades:
            batch.avg_grade = sum(grades) / len(grades)
            batch.min_grade = min(grades)
            batch.max_grade = max(grades)
        
        db.session.commit()
        
        return json_response({
            'batch_id': batch.batch_id,
            'sample_count': len(sample_results),
            'avg_grade': batch.avg_grade,
            'min_grade': batch.min_grade,
            'max_grade': batch.max_grade,
            'samples': sample_results
        })
        
    except Exception as e:
        db.session.rollback()
        return error_response(str(e), 500)
