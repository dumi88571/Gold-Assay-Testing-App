"""Calculator tools routes."""

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required

from app.schemas import CalculationRequest, CalculationResponse
from app.utils.helpers import json_response, error_response

bp = Blueprint('tools', __name__)


@bp.route('/')
@login_required
def index():
    """Tools dashboard."""
    return render_template('tools.html')


@bp.route('/api/calculate', methods=['POST'])
@login_required
def api_calculate():
    """API endpoint for various calculations."""
    try:
        data = request.get_json()
        calc_request = CalculationRequest(**data)
        
        calc_type = calc_request.type
        result = 0.0
        unit = ''
        formula = ''
        
        if calc_type == 'moisture':
            if calc_request.wet_weight and calc_request.dry_weight:
                if calc_request.wet_weight > 0:
                    result = ((calc_request.wet_weight - calc_request.dry_weight) 
                             / calc_request.wet_weight) * 100
                unit = '%'
                formula = '((wet - dry) / wet) × 100'
            else:
                return error_response('Wet and dry weights required')
        
        elif calc_type == 'pulp_density':
            if calc_request.sg_dry and calc_request.sg_pulp:
                if calc_request.sg_pulp > 1:
                    result = 100 * (calc_request.sg_dry / (calc_request.sg_dry - 1)) * \
                            (1 - (1 / calc_request.sg_pulp))
                unit = '% Solids'
                formula = '100 × (SG_dry/(SG_dry-1)) × (1 - 1/SG_pulp)'
            else:
                return error_response('SG dry and SG pulp required')
        
        elif calc_type == 'fineness':
            if calc_request.gold_weight and calc_request.total_weight:
                if calc_request.total_weight > 0:
                    result = (calc_request.gold_weight / calc_request.total_weight) * 1000
                unit = '‰'
                formula = '(gold / total) × 1000'
            else:
                return error_response('Gold and total weights required')
        
        elif calc_type == 'weighted_avg':
            if calc_request.samples:
                total_weight = sum(s.get('weight', 0) for s in calc_request.samples)
                total_value = sum(s.get('weight', 0) * s.get('grade', 0) 
                                 for s in calc_request.samples)
                if total_weight > 0:
                    result = total_value / total_weight
                unit = 'g/t'
                formula = 'Σ(weight × grade) / Σ(weight)'
            else:
                return error_response('Samples list required')
        
        elif calc_type == 'recovery':
            if (calc_request.feed_grade is not None and 
                calc_request.concentrate_grade is not None and
                calc_request.tailings_grade is not None):
                
                # Two-product formula
                if calc_request.concentrate_grade != calc_request.tailings_grade:
                    result = ((calc_request.feed_grade - calc_request.tailings_grade) / 
                             (calc_request.concentrate_grade - calc_request.tailings_grade)) * 100
                unit = '%'
                formula = '((feed - tail) / (conc - tail)) × 100'
            else:
                return error_response('Feed, concentrate, and tailings grades required')
        
        elif calc_type == 'concentration_ratio':
            if (calc_request.feed_grade is not None and 
                calc_request.concentrate_grade is not None):
                
                if calc_request.feed_grade > 0:
                    result = calc_request.concentrate_grade / calc_request.feed_grade
                unit = ':1'
                formula = 'concentrate / feed'
            else:
                return error_response('Feed and concentrate grades required')
        
        return json_response({
            'result': round(result, 4),
            'unit': unit,
            'formula_used': formula
        })
        
    except Exception as e:
        return error_response(str(e), 400)


@bp.route('/moisture')
@login_required
def moisture_calculator():
    """Moisture content calculator page."""
    return render_template('tool_moisture.html')


@bp.route('/pulp-density')
@login_required
def pulp_density_calculator():
    """Pulp density calculator page."""
    return render_template('tool_pulp_density.html')


@bp.route('/fineness')
@login_required
def fineness_calculator():
    """Bullion fineness calculator page."""
    return render_template('tool_fineness.html')


@bp.route('/weighted-avg')
@login_required
def weighted_avg_calculator():
    """Weighted average calculator page."""
    return render_template('tool_weighted_avg.html')


@bp.route('/recovery')
@login_required
def recovery_calculator():
    """Recovery calculator page."""
    return render_template('tool_recovery.html')
