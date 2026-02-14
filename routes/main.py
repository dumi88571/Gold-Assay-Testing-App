"""Main routes for the application."""

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from sqlalchemy import func
from datetime import datetime, timedelta

from app import db
from app.models import Sample, Batch, AuditLog
from app.services.ml_engine import get_ml_engine

bp = Blueprint('main', __name__)


@bp.route('/')
@login_required
def index():
    """Main dashboard view."""
    # Get statistics
    total_samples = Sample.query.count()
    recent_samples = Sample.query.order_by(Sample.created_at.desc()).limit(20).all()
    
    # Calculate 7-day trend
    week_ago = datetime.utcnow() - timedelta(days=7)
    daily_counts = db.session.query(
        func.date(Sample.created_at).label('date'),
        func.count(Sample.id).label('count')
    ).filter(Sample.created_at >= week_ago).group_by(func.date(Sample.created_at)).all()
    
    # Convert to dict for JSON serialization
    daily_counts = [{'date': str(d.date), 'count': d.count} for d in daily_counts]
    
    # Grade statistics for recent samples
    recent_grades = [s.predicted_grade for s in recent_samples if s.predicted_grade]
    avg_grade = sum(recent_grades) / len(recent_grades) if recent_grades else 0
    
    # Get ML engine status
    ml_engine = get_ml_engine()
    
    # QC statistics
    qc_passed = Sample.query.filter_by(qc_passed=True).count()
    qc_failed = Sample.query.filter_by(qc_passed=False).count()
    
    # Batch statistics
    active_batches = Batch.query.filter_by(status='pending').count()
    completed_batches = Batch.query.filter_by(status='completed').count()
    
    return render_template('dashboard.html',
                         total_samples=total_samples,
                         recent_samples=recent_samples,
                         daily_counts=daily_counts,
                         avg_grade=avg_grade,
                         ml_accuracy=ml_engine.metrics.get('r2_score', 0.0) * 100,
                         ml_version=ml_engine.model_version,
                         qc_passed=qc_passed,
                         qc_failed=qc_failed,
                         active_batches=active_batches,
                         completed_batches=completed_batches)


@bp.route('/history')
@login_required
def history():
    """View sample history with filtering."""
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    # Build query with filters
    query = Sample.query
    
    if request.args.get('sample_type'):
        query = query.filter_by(sample_type=request.args.get('sample_type'))
    
    if request.args.get('assay_method'):
        query = query.filter_by(assay_method=request.args.get('assay_method'))
    
    if request.args.get('date_from'):
        query = query.filter(Sample.created_at >= request.args.get('date_from'))
    
    if request.args.get('date_to'):
        query = query.filter(Sample.created_at <= request.args.get('date_to'))
    
    if request.args.get('qc_passed') == 'true':
        query = query.filter_by(qc_passed=True)
    elif request.args.get('qc_passed') == 'false':
        query = query.filter_by(qc_passed=False)
    
    # Get paginated results
    samples = query.order_by(Sample.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('history.html', samples=samples)


@bp.route('/sample/<sample_id>')
@login_required
def sample_detail(sample_id):
    """View detailed sample information."""
    sample = Sample.query.filter_by(sample_id=sample_id).first_or_404()
    
    # Get audit history for this sample
    audit_logs = AuditLog.query.filter_by(
        entity_type='Sample',
        entity_id=sample_id
    ).order_by(AuditLog.timestamp.desc()).all()
    
    # Generate explanation
    ml_engine = get_ml_engine()
    
    # Reconstruct input data
    input_data = {
        'sample_type': sample.sample_type,
        'assay_method': sample.assay_method,
    }
    for feature in ml_engine.FEATURES:
        input_data[feature] = getattr(sample, feature, 0.0)
        
    explanation = ml_engine.explain_prediction(input_data)
    
    return render_template('sample_detail.html',
                         sample=sample,
                         audit_logs=audit_logs,
                         explanation=explanation)


@bp.route('/certificate/<sample_id>')
@login_required
def certificate(sample_id):
    """Generate analysis certificate for printing."""
    sample = Sample.query.filter_by(sample_id=sample_id).first_or_404()
    return render_template('certificate.html', sample=sample)


@bp.route('/search')
@login_required
def search():
    """Search samples by various criteria."""
    q = request.args.get('q', '')
    
    if not q:
        return render_template('search.html', results=None, query='')
    
    # Search in sample IDs and other fields
    results = Sample.query.filter(
        db.or_(
            Sample.sample_id.ilike(f'%{q}%'),
            Sample.client_name.ilike(f'%{q}%'),
            Sample.sample_type.ilike(f'%{q}%')
        )
    ).order_by(Sample.created_at.desc()).limit(100).all()
    
    return render_template('search.html', results=results, query=q)


@bp.route('/api/dashboard-stats')
@login_required
def dashboard_stats():
    """API endpoint for dashboard statistics."""
    # Time range
    days = request.args.get('days', 30, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    # Sample counts by type
    type_counts = db.session.query(
        Sample.sample_type,
        func.count(Sample.id)
    ).filter(Sample.created_at >= since).group_by(Sample.sample_type).all()
    
    # Grade trend (daily averages)
    grade_trend = db.session.query(
        func.date(Sample.created_at).label('date'),
        func.avg(Sample.predicted_grade).label('avg_grade'),
        func.count(Sample.id).label('count')
    ).filter(Sample.created_at >= since).group_by(
        func.date(Sample.created_at)
    ).order_by('date').all()
    
    # Method distribution
    method_counts = db.session.query(
        Sample.assay_method,
        func.count(Sample.id)
    ).filter(Sample.created_at >= since).group_by(Sample.assay_method).all()
    
    return jsonify({
        'sample_types': [{'type': t, 'count': c} for t, c in type_counts],
        'grade_trend': [
            {'date': str(d), 'avg_grade': float(g), 'count': c} 
            for d, g, c in grade_trend
        ],
        'methods': [{'method': m, 'count': c} for m, c in method_counts]
    })
