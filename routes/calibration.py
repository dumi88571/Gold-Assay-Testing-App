"""Calibration management routes."""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from pydantic import ValidationError
from datetime import datetime, timedelta

from app import db
from app.models import CalibrationRecord, CalibrationStandard, Sample
from app.schemas import CalibrationInput
from app.utils.helpers import require_role

bp = Blueprint('calibration', __name__)


@bp.route('/')
@login_required
def index():
    """Calibration management dashboard."""
    # Recent calibrations
    calibrations = CalibrationRecord.query.order_by(
        CalibrationRecord.calibration_date.desc()
    ).limit(20).all()
    
    # Available standards
    standards = CalibrationStandard.query.filter_by(is_active=True).all()
    
    # Status summary
    last_calibration = CalibrationRecord.query.filter_by(passed=True).order_by(
        CalibrationRecord.calibration_date.desc()
    ).first()
    
    days_since = None
    if last_calibration:
        days_since = (datetime.utcnow() - last_calibration.calibration_date).days
    
    return render_template('calibration.html',
                         calibrations=calibrations,
                         standards=standards,
                         last_calibration=last_calibration,
                         days_since=days_since)


@bp.route('/add', methods=['POST'])
@login_required
@require_role('analyst')
def add():
    """Add new calibration record."""
    try:
        data = request.form.to_dict()
        calibration_input = CalibrationInput(**data)
        
        record = CalibrationRecord(
            instrument_id=calibration_input.instrument_id,
            analyst_id=current_user.id,
            calibration_type=calibration_input.calibration_type,
            standard_used=calibration_input.standard_used,
            drift_check=calibration_input.drift_check,
            slope=calibration_input.slope,
            intercept=calibration_input.intercept,
            r_squared=calibration_input.r_squared,
            notes=calibration_input.notes
        )
        
        # Check if passed
        record.passed = (
            calibration_input.r_squared >= 0.995 and
            abs(calibration_input.drift_check - 1.0) <= 0.05 and
            abs(calibration_input.slope - 1.0) <= 0.1
        )
        
        db.session.add(record)
        db.session.commit()
        
        status = 'passed' if record.passed else 'failed'
        flash(f'Calibration record added. Status: {status}', 
              'success' if record.passed else 'warning')
        
        return redirect(url_for('calibration.index'))
        
    except ValidationError as e:
        flash(f'Validation error: {e.errors()}', 'danger')
        return redirect(url_for('calibration.index'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding calibration: {str(e)}', 'danger')
        return redirect(url_for('calibration.index'))


@bp.route('/history')
@login_required
def history():
    """View calibration history."""
    instrument = request.args.get('instrument')
    days = request.args.get('days', 90, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    query = CalibrationRecord.query.filter(
        CalibrationRecord.calibration_date >= since
    )
    
    if instrument:
        query = query.filter_by(instrument_id=instrument)
    
    calibrations = query.order_by(
        CalibrationRecord.calibration_date.desc()
    ).all()
    
    return render_template('calibration_history.html',
                         calibrations=calibrations,
                         instrument=instrument,
                         days=days)


@bp.route('/trends')
@login_required
def trends():
    """View calibration trends over time."""
    instrument_id = request.args.get('instrument')
    days = request.args.get('days', 90, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    query = CalibrationRecord.query.filter(
        CalibrationRecord.calibration_date >= since
    )
    
    if instrument_id:
        query = query.filter_by(instrument_id=instrument_id)
    
    records = query.order_by(CalibrationRecord.calibration_date).all()
    
    trend_data = []
    for r in records:
        trend_data.append({
            'date': r.calibration_date.isoformat(),
            'drift': r.drift_check,
            'slope': r.slope,
            'r_squared': r.r_squared,
            'passed': r.passed
        })
    
    return render_template('calibration_trends.html',
                         trend_data=trend_data,
                         instrument_id=instrument_id,
                         days=days)


@bp.route('/api/calibration-status')
@login_required
def api_status():
    """Get current calibration status."""
    instruments = db.session.query(
        CalibrationRecord.instrument_id
    ).distinct().all()
    
    status = {}
    for (inst,) in instruments:
        last = CalibrationRecord.query.filter_by(
            instrument_id=inst
        ).order_by(CalibrationRecord.calibration_date.desc()).first()
        
        if last:
            days_since = (datetime.utcnow() - last.calibration_date).days
            status[inst] = {
                'last_calibration': last.calibration_date.isoformat(),
                'days_since': days_since,
                'status': 'current' if days_since <= 1 else 'stale',
                'passed': last.passed,
                'drift': last.drift_check,
                'r_squared': last.r_squared
            }
    
    return jsonify(status)


@bp.route('/standards')
@login_required
def standards():
    """Manage calibration standards."""
    standards = CalibrationStandard.query.all()
    return render_template('standards.html', standards=standards)
