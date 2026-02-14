"""Utility functions for Gold Assay Analyzer."""

import hashlib
import io
import csv
import json
from datetime import datetime, date
from decimal import Decimal
from functools import wraps
from flask import jsonify, request
from typing import Dict, Any, List


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


def json_response(data: Any, status_code: int = 200) -> jsonify:
    """Create a standardized JSON response."""
    response = {
        'status': 'success' if status_code < 400 else 'error',
        'timestamp': datetime.utcnow().isoformat(),
        'data': data
    }
    return jsonify(response), status_code


def error_response(message: str, status_code: int = 400, details: Dict = None) -> jsonify:
    """Create a standardized error response."""
    response = {
        'status': 'error',
        'timestamp': datetime.utcnow().isoformat(),
        'message': message,
        'details': details or {}
    }
    return jsonify(response), status_code


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistics for a list of values."""
    if not values:
        return {}
    
    import numpy as np
    
    arr = np.array(values)
    
    return {
        'count': len(arr),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'var': float(np.var(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
        'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        'cv': float(np.std(arr) / np.mean(arr) * 100) if np.mean(arr) != 0 else 0
    }


def calculate_grade_distribution(values: List[float], bins: int = 10) -> List[Dict]:
    """Calculate grade distribution histogram."""
    if not values:
        return []
    
    import numpy as np
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        return [{'range': f'{min_val:.2f}', 'count': len(values), 'percentage': 100.0}]
    
    hist, edges = np.histogram(values, bins=bins)
    total = len(values)
    
    distribution = []
    for i, count in enumerate(hist):
        distribution.append({
            'range': f'{edges[i]:.2f} - {edges[i+1]:.2f}',
            'min': float(edges[i]),
            'max': float(edges[i+1]),
            'count': int(count),
            'percentage': round(count / total * 100, 2)
        })
    
    return distribution


def calculate_correlation_matrix(data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Calculate correlation matrix between features."""
    import numpy as np
    
    keys = list(data.keys())
    matrix = {}
    
    for i, key1 in enumerate(keys):
        matrix[key1] = {}
        for j, key2 in enumerate(keys):
            if len(data[key1]) == len(data[key2]) and len(data[key1]) > 1:
                corr = np.corrcoef(data[key1], data[key2])[0, 1]
                matrix[key1][key2] = float(corr)
            else:
                matrix[key1][key2] = 0.0
    
    return matrix


def export_to_csv(samples: List[Any], include_fields: List[str] = None) -> io.StringIO:
    """Export samples to CSV format."""
    if include_fields is None:
        include_fields = [
            'sample_id', 'created_at', 'sample_type', 'assay_method',
            'predicted_grade', 'actual_grade', 'model_confidence', 'uncertainty',
            'status', 'qc_passed'
        ]
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(include_fields)
    
    # Write data
    for sample in samples:
        row = []
        for field in include_fields:
            value = getattr(sample, field, None)
            if value is None:
                value = ''
            elif isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            row.append(value)
        writer.writerow(row)
    
    output.seek(0)
    return output


def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Validate file extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def compute_file_hash(file_stream) -> str:
    """Compute MD5 hash of a file stream."""
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: file_stream.read(4096), b""):
        hash_md5.update(chunk)
    file_stream.seek(0)
    return hash_md5.hexdigest()


def parse_csv_batch(file_stream) -> List[Dict]:
    """Parse CSV file for batch processing."""
    content = file_stream.read().decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def parse_excel_batch(file_stream) -> List[Dict]:
    """Parse Excel file for batch processing."""
    try:
        import pandas as pd
        df = pd.read_excel(file_stream)
        return df.to_dict('records')
    except ImportError:
        raise ValueError("pandas required for Excel parsing")


def log_audit_action(action: str, entity_type: str = None, entity_id: Any = None,
                    old_values: Dict = None, new_values: Dict = None, notes: str = None):
    """Log an action to the audit trail."""
    from app.models import AuditLog
    from flask_login import current_user
    
    AuditLog.log_action(
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        old_values=old_values,
        new_values=new_values,
        notes=notes
    )


def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask_login import current_user
        if not current_user.is_authenticated:
            return error_response('Authentication required', 401)
        return f(*args, **kwargs)
    return decorated_function


def require_role(role: str):
    """Decorator factory to require specific role."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask_login import current_user
            if not current_user.is_authenticated:
                return error_response('Authentication required', 401)
            if not current_user.has_role(role):
                return error_response(f'Requires {role} role', 403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
