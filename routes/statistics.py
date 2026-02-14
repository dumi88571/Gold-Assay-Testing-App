"""Statistical analysis routes."""

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from sqlalchemy import func
from datetime import datetime, timedelta
import numpy as np

from app import db
from app.models import Sample
from app.utils.helpers import calculate_statistics, calculate_grade_distribution, calculate_correlation_matrix

bp = Blueprint('statistics', __name__)


@bp.route('/')
@login_required
def index():
    """Statistics dashboard."""
    # Date range
    days = request.args.get('days', 30, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    # Get samples in range
    samples = Sample.query.filter(
        Sample.created_at >= since,
        Sample.predicted_grade.isnot(None)
    ).all()
    
    grades = [s.predicted_grade for s in samples]
    
    stats = calculate_statistics(grades) if grades else {}
    distribution = calculate_grade_distribution(grades, bins=10) if grades else []
    
    # Feature correlation data
    feature_data = {}
    for feature in [
        'absorption_242nm', 'absorption_267nm', 'emission_intensity',
        'iron_ppm', 'copper_ppm', 'silver_ppm'
    ]:
        values = [getattr(s, feature) for s in samples if getattr(s, feature) is not None]
        feature_data[feature] = values
    feature_data['predicted_grade'] = grades
    
    correlation_matrix = calculate_correlation_matrix(feature_data)
    
    return render_template('statistics.html',
                         stats=stats,
                         distribution=distribution,
                         correlation_matrix=correlation_matrix,
                         sample_count=len(samples),
                         days=days)


@bp.route('/outliers')
@login_required
def outliers():
    """Identify outlier samples."""
    days = request.args.get('days', 30, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    samples = Sample.query.filter(
        Sample.created_at >= since,
        Sample.predicted_grade.isnot(None)
    ).all()
    
    if not samples:
        return render_template('outliers.html', outliers=[], method='iqr')
    
    grades = [s.predicted_grade for s in samples]
    
    # IQR method for outlier detection
    q1 = np.percentile(grades, 25)
    q3 = np.percentile(grades, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [s for s in samples if s.predicted_grade < lower_bound or s.predicted_grade > upper_bound]
    
    return render_template('outliers.html',
                         outliers=outliers,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound,
                         method='iqr',
                         sample_count=len(samples))


@bp.route('/trends')
@login_required
def trends():
    """Trend analysis over time."""
    days = request.args.get('days', 90, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    # Daily statistics
    daily_stats = db.session.query(
        func.date(Sample.created_at).label('date'),
        func.avg(Sample.predicted_grade).label('avg'),
        func.min(Sample.predicted_grade).label('min'),
        func.max(Sample.predicted_grade).label('max'),
        func.count(Sample.id).label('count')
    ).filter(
        Sample.created_at >= since,
        Sample.predicted_grade.isnot(None)
    ).group_by(
        func.date(Sample.created_at)
    ).order_by('date').all()
    
    trend_data = []
    for stat in daily_stats:
        trend_data.append({
            'date': str(stat.date),
            'avg': float(stat.avg) if stat.avg else 0,
            'min': float(stat.min) if stat.min else 0,
            'max': float(stat.max) if stat.max else 0,
            'count': stat.count
        })
    
    # Moving average (7-day)
    if len(trend_data) >= 7:
        for i in range(len(trend_data)):
            if i >= 6:
                window = trend_data[i-6:i+1]
                ma = sum(d['avg'] for d in window) / 7
                trend_data[i]['moving_avg'] = round(ma, 3)
    
    return render_template('trends.html', trend_data=trend_data, days=days)


@bp.route('/correlations')
@login_required
def correlations():
    """Feature correlation analysis."""
    days = request.args.get('days', 30, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    # Get samples with all features
    samples = Sample.query.filter(
        Sample.created_at >= since,
        Sample.predicted_grade.isnot(None),
        Sample.absorption_242nm.isnot(None),
        Sample.iron_ppm.isnot(None)
    ).limit(500).all()
    
    features = [
        'absorption_242nm', 'absorption_267nm', 'emission_intensity',
        'solution_ph', 'temperature_c', 'ionic_strength',
        'iron_ppm', 'copper_ppm', 'silver_ppm', 'sulfur_content',
        'calibration_drift', 'replicate_rsd', 'blank_intensity',
        'predicted_grade'
    ]
    
    feature_data = {}
    for f in features:
        values = [getattr(s, f) for s in samples if getattr(s, f) is not None]
        if values:
            feature_data[f] = values
    
    matrix = calculate_correlation_matrix(feature_data)
    
    return render_template('correlations.html', 
                         matrix=matrix, 
                         features=list(feature_data.keys()),
                         sample_count=len(samples))


@bp.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for statistics."""
    metric = request.args.get('metric', 'summary')
    days = request.args.get('days', 30, type=int)
    since = datetime.utcnow() - timedelta(days=days)
    
    samples = Sample.query.filter(
        Sample.created_at >= since,
        Sample.predicted_grade.isnot(None)
    ).all()
    
    grades = [s.predicted_grade for s in samples]
    
    if metric == 'summary':
        return jsonify(calculate_statistics(grades))
    
    elif metric == 'distribution':
        bins = request.args.get('bins', 10, type=int)
        return jsonify(calculate_grade_distribution(grades, bins))
    
    elif metric == 'by_type':
        by_type = {}
        for sample_type in ['Concentrate', 'Tailings', 'Solution', 'Feed']:
            type_grades = [s.predicted_grade for s in samples if s.sample_type == sample_type]
            if type_grades:
                by_type[sample_type] = calculate_statistics(type_grades)
        return jsonify(by_type)
    
    return jsonify({'error': 'Unknown metric'}), 400
