"""REST API routes."""

from flask import Blueprint, request, jsonify, Response
from flask_login import login_required, current_user
from sqlalchemy import func
from datetime import datetime, timedelta
import io
import csv

from app import db
from app.models import Sample, Batch, AuditLog, CalibrationRecord
from app.schemas import ExportRequest
from app.utils.helpers import (
    json_response, error_response, export_to_csv,
    calculate_statistics, calculate_grade_distribution
)

bp = Blueprint('api', __name__)


@bp.route('/samples', methods=['GET'])
@login_required
def list_samples():
    """List samples with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    # Filters
    query = Sample.query
    
    if request.args.get('sample_type'):
        query = query.filter_by(sample_type=request.args.get('sample_type'))
    
    if request.args.get('batch_id'):
        query = query.filter_by(batch_id=request.args.get('batch_id'))
    
    if request.args.get('date_from'):
        query = query.filter(Sample.created_at >= request.args.get('date_from'))
    
    if request.args.get('date_to'):
        query = query.filter(Sample.created_at <= request.args.get('date_to'))
    
    # Sort by creation date descending
    query = query.order_by(Sample.created_at.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return json_response({
        'samples': [s.to_dict() for s in pagination.items],
        'pagination': {
            'page': pagination.page,
            'per_page': pagination.per_page,
            'total': pagination.total,
            'pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    })


@bp.route('/samples/<sample_id>')
@login_required
def get_sample(sample_id):
    """Get single sample details."""
    sample = Sample.query.filter_by(sample_id=sample_id).first_or_404()
    return json_response(sample.to_dict())


@bp.route('/samples/<sample_id>', methods=['PUT'])
@login_required
def update_sample(sample_id):
    """Update sample data."""
    sample = Sample.query.filter_by(sample_id=sample_id).first_or_404()
    
    data = request.get_json()
    old_values = sample.to_dict()
    
    # Update allowed fields
    if 'actual_grade' in data:
        sample.actual_grade = data['actual_grade']
        sample.status = 'validated'
    
    if 'notes' in data:
        sample.notes = data['notes']
    
    if 'status' in data:
        sample.status = data['status']
    
    db.session.commit()
    
    AuditLog.log_action(
        action='update',
        entity_type='Sample',
        entity_id=sample_id,
        old_values=old_values,
        new_values=sample.to_dict()
    )
    
    return json_response(sample.to_dict())


@bp.route('/samples/<sample_id>', methods=['DELETE'])
@login_required
def delete_sample(sample_id):
    """Delete sample (soft delete by marking as rejected)."""
    sample = Sample.query.filter_by(sample_id=sample_id).first_or_404()
    
    old_values = sample.to_dict()
    sample.status = 'rejected'
    
    db.session.commit()
    
    AuditLog.log_action(
        action='delete',
        entity_type='Sample',
        entity_id=sample_id,
        old_values=old_values,
        notes='Sample marked as rejected'
    )
    
    return json_response({'message': 'Sample deleted', 'sample_id': sample_id})


@bp.route('/batches')
@login_required
def list_batches():
    """List batches."""
    batches = Batch.query.order_by(Batch.created_at.desc()).all()
    return json_response([{
        'batch_id': b.batch_id,
        'batch_name': b.batch_name,
        'status': b.status,
        'sample_count': b.sample_count,
        'avg_grade': b.avg_grade,
        'created_at': b.created_at.isoformat() if b.created_at else None
    } for b in batches])


@bp.route('/batches/<batch_id>')
@login_required
def get_batch(batch_id):
    """Get batch with samples."""
    batch = Batch.query.filter_by(batch_id=batch_id).first_or_404()
    samples = Sample.query.filter_by(batch_id=batch_id).all()
    
    return json_response({
        'batch_id': batch.batch_id,
        'batch_name': batch.batch_name,
        'status': batch.status,
        'sample_count': batch.sample_count,
        'completed_count': batch.completed_count,
        'failed_count': batch.failed_count,
        'avg_grade': batch.avg_grade,
        'min_grade': batch.min_grade,
        'max_grade': batch.max_grade,
        'samples': [s.to_dict() for s in samples]
    })


@bp.route('/export', methods=['POST'])
@login_required
def export_data():
    """Export data in various formats."""
    try:
        data = request.get_json() or {}
        export_req = ExportRequest(**data)
        
        # Build query
        query = Sample.query
        
        if export_req.date_from:
            query = query.filter(Sample.created_at >= export_req.date_from)
        
        if export_req.date_to:
            query = query.filter(Sample.created_at <= export_req.date_to)
        
        if export_req.sample_type:
            query = query.filter_by(sample_type=export_req.sample_type)
        
        samples = query.order_by(Sample.created_at.desc()).limit(10000).all()
        
        if export_req.format == 'csv':
            output = export_to_csv(samples, export_req.include_fields)
            
            response = Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=assay_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                }
            )
            return response
        
        elif export_req.format == 'json':
            return json_response([s.to_dict() for s in samples])
        
        else:
            return error_response(f'Unsupported format: {export_req.format}')
    
    except Exception as e:
        return error_response(str(e), 500)


@bp.route('/audit-logs')
@login_required
def audit_logs():
    """Get audit logs."""
    entity_type = request.args.get('entity_type')
    entity_id = request.args.get('entity_id')
    days = request.args.get('days', 30, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    query = AuditLog.query.filter(AuditLog.timestamp >= since)
    
    if entity_type:
        query = query.filter_by(entity_type=entity_type)
    
    if entity_id:
        query = query.filter_by(entity_id=entity_id)
    
    logs = query.order_by(AuditLog.timestamp.desc()).limit(500).all()
    
    return json_response([{
        'id': log.id,
        'timestamp': log.timestamp.isoformat() if log.timestamp else None,
        'action': log.action,
        'entity_type': log.entity_type,
        'entity_id': log.entity_id,
        'username': log.username,
        'notes': log.notes
    } for log in logs])


@bp.route('/health')
def health_check():
    """Health check endpoint."""
    from app.services.ml_engine import get_ml_engine
    
    ml_engine = get_ml_engine()
    
    return json_response({
        'status': 'healthy',
        'database': 'connected',
        'ml_model': {
            'loaded': ml_engine.model is not None,
            'accuracy': ml_engine.accuracy,
            'version': ml_engine.model_version,
            'last_trained': ml_engine.last_trained.isoformat() if ml_engine.last_trained else None
        },
        'timestamp': datetime.utcnow().isoformat()
    })
