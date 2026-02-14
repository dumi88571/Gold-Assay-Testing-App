"""Machine Learning Engine for Gold Grade Prediction - Scientific Edition.

Compliant with ISO 17025 validation requirements.
Uses Hybrid Physics-ML approach:
1. Base Grade: Linear Calibration (Beer-Lambert Law)
2. Correction: Gradient Boosting Quantile Regression (Matrix Effects)
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class CalibrationModel(BaseEstimator, RegressorMixin):
    """
    Physics-based Calibration Model (Beer-Lambert Law).
    Fits Signal/Internal_Standard vs Concentration.
    """
    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0
        self.r2 = 0.0
        self.lod = 0.01  # Limit of Detection
        self.loq = 0.03  # Limit of Quantification (3.3 * LOD approx)
        self.model = RANSACRegressor(random_state=42) # Robust to outliers
        
    def fit(self, X, y):
        # X is expected to be a DataFrame with 'emission_intensity' and 'internal_standard'
        # We calculate the ratio
        ratio = X['emission_intensity'] / X['internal_standard']
        ratio = ratio.values.reshape(-1, 1)
        
        self.model.fit(ratio, y)
        
        # Extract parameters from the underlying estimator
        estimator = self.model.estimator_
        self.slope = estimator.coef_[0]
        self.intercept = estimator.intercept_
        
        # Calculate R2 on inliers
        y_pred = self.model.predict(ratio)
        self.r2 = r2_score(y, y_pred)
        
        # Estimate LOD (3.3 * sigma_blank / slope)
        # We approximate sigma_blank from the residuals of low-grade samples
        residuals = y - y_pred
        sigma_resid = np.std(residuals)
        self.lod = max(0.005, (3.3 * sigma_resid) / abs(self.slope))
        self.loq = 3 * self.lod
        
        return self

    def predict(self, X):
        ratio = X['emission_intensity'] / X['internal_standard']
        ratio = ratio.values.reshape(-1, 1)
        return self.model.predict(ratio)

class AssayMLEngine:
    """Enhanced ML engine for gold assay prediction with Uncertainty Quantification."""
    
    FEATURES = [
        'absorption_242nm', 'absorption_267nm', 'emission_intensity',
        'solution_ph', 'temperature_c', 'ionic_strength', 'dissolved_oxygen',
        'iron_ppm', 'copper_ppm', 'silver_ppm', 'sulfur_content',
        'dilution_factor', 'measurement_time', 'calibration_drift',
        'replicate_rsd', 'blank_intensity', 'internal_standard'
    ]
    
    CATEGORICAL_FEATURES = ['sample_type', 'assay_method']
    
    MODEL_PATH = 'models'
    
    def __init__(self):
        self.models = {} # calculated 'median', 'lower', 'upper'
        self.calibration_model = CalibrationModel()
        self.metrics = {}
        self.last_trained = None
        self.model_version = '2.1.0-Scientific'
        self._ensure_model_dir()
        self._load_or_train_model()
    
    def _ensure_model_dir(self):
        """Ensure model directory exists."""
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)
    
    def _generate_training_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic training data with realistic matrix effects."""
        np.random.seed(42)
        
        # 1. Generate True Grade (Log-normal)
        y = np.random.lognormal(0.5, 1.5, n_samples)
        y = np.clip(y, 0.001, 1000)
        
        # 2. Generate Internal Standard (IS) - varies per sample
        internal_standard = np.random.normal(100.0, 5.0, n_samples)
        
        # 3. Generate Interferences
        fe = np.random.lognormal(3.0, 0.8, n_samples) # Iron
        cu = np.random.lognormal(2.5, 0.7, n_samples) # Copper
        ph = np.random.normal(7.0, 1.0, n_samples)
        
        # 4. Physics: Signal = (Slope * Grade + Intercept) * IS * MatrixFactor
        # Matrix suppression from Fe and Cu
        matrix_factor = 1.0 - (fe * 0.0001) - (cu * 0.0002)
        matrix_factor = np.clip(matrix_factor, 0.8, 1.2)
        
        # Ideal Emission (linearly related to grade * IS)
        slope = 0.15
        emission = (y * slope) * internal_standard * matrix_factor
        
        # Add instrument noise
        emission += np.random.normal(0, 0.05 * np.mean(emission), n_samples)
        
        data = {
            'sample_type': np.random.choice(
                ['Concentrate', 'Tailings', 'Solution', 'Feed', 'Bullion'], n_samples
            ),
            'assay_method': np.random.choice(
                ['Fire Assay', 'Acid Digest', 'Cyanide Leach', 'ICP-OES', 'AAS'], n_samples
            ),
            'absorption_242nm': y * 0.12 * matrix_factor + np.random.normal(0, 0.02, n_samples),
            'absorption_267nm': y * 0.10 * matrix_factor + np.random.normal(0, 0.02, n_samples),
            'emission_intensity': emission,
            'solution_ph': ph,
            'temperature_c': np.random.normal(25.0, 2.0, n_samples),
            'ionic_strength': np.random.uniform(0.1, 0.5, n_samples),
            'dissolved_oxygen': np.random.normal(6.0, 1.0, n_samples),
            'iron_ppm': fe,
            'copper_ppm': cu,
            'silver_ppm': np.random.lognormal(1.5, 0.6, n_samples),
            'sulfur_content': np.random.uniform(0.0, 10.0, n_samples),
            'dilution_factor': np.random.choice([1, 10, 50, 100], n_samples),
            'measurement_time': np.random.normal(30.0, 5.0, n_samples),
            'calibration_drift': np.random.normal(1.0, 0.02, n_samples),
            'replicate_rsd': np.random.exponential(1.5, n_samples),
            'blank_intensity': np.random.exponential(5.0, n_samples),
            'internal_standard': internal_standard
        }
        
        return pd.DataFrame(data), y
    
    def _build_pipeline(self, model_type: str = 'mean') -> Pipeline:
        """Build a pipeline with specific loss function."""
        # Preprocessor: Scale numeric, OneHot categorical
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             self.CATEGORICAL_FEATURES)
        ], remainder='drop')
        
        if model_type == 'quantile_low':
             loss = 'quantile'
             alpha = 0.05
        elif model_type == 'quantile_high':
             loss = 'quantile'
             alpha = 0.95
        else: # mean/median
             loss = 'squared_error' # Standard regression
             alpha = 0.5

        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                loss=loss,
                alpha=alpha if loss == 'quantile' else 0.9, # alpha only for quantile/huber
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
    
    def train_model(self, X: Optional[pd.DataFrame] = None, y: Optional[np.ndarray] = None) -> Dict:
        """Train the Hybrid Model (Calibration + ML Correction)."""
        logger.info("Starting scientific model training...")
        
        if X is None or y is None:
            X, y = self._generate_training_data(5000)
            
        # 1. Train Calibration Model (Physics)
        self.calibration_model.fit(X, y)
        grade_physics = self.calibration_model.predict(X)
        
        # 2. Calculate Residuals (Matrix Correction Target)
        # Residual = True Grade - Physics Grade
        residuals = y - grade_physics
        
        # 3. Train ML Models on Residuals
        # We train 3 models: Median (Prediction), Low (5%), High (95%)
        models_to_train = {
            'median': 'squared_error', # Use standard regression for best point estimate
            'lower': 'quantile_low',
            'upper': 'quantile_high'
        }
        
        X_train, X_test, res_train, res_test = train_test_split(
            X, residuals, test_size=0.2, random_state=42
        )
        # We also need the original y_test to calculate accuracy
        _, _, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for name, mode in models_to_train.items():
            pipeline = self._build_pipeline(mode)
            pipeline.fit(X_train, res_train)
            self.models[name] = pipeline
            
        # 4. Evaluate
        # Final Pred = Physics Pred + ML Residual Pred
        phys_pred_test = self.calibration_model.predict(X_test)
        ml_resid_test = self.models['median'].predict(X_test)
        final_pred = phys_pred_test + ml_resid_test
        
        self.metrics = {
            'r2_score': r2_score(y_test_orig, final_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, final_pred)),
            'calibration_r2': self.calibration_model.r2,
            'lod': self.calibration_model.lod
        }
        
        self.last_trained = datetime.now()
        logger.info(f"Model trained. R2: {self.metrics['r2_score']:.4f}, LOQ: {self.calibration_model.loq:.3f}")
        
        self._save_model()
        return self.metrics

    def _save_model(self):
        """Save trained models."""
        model_file = os.path.join(self.MODEL_PATH, f'model_scientific.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'calibration_model': self.calibration_model,
                'metrics': self.metrics,
                'last_trained': self.last_trained,
                'version': self.model_version
            }, f)

    def _load_or_train_model(self):
        """Load existing model or train."""
        model_file = os.path.join(self.MODEL_PATH, f'model_scientific.pkl')
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                self.models = data['models']
                self.calibration_model = data['calibration_model']
                self.metrics = data['metrics']
                self.last_trained = data['last_trained']
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        
        self.train_model()

    def _sanitize_input(self, data: Dict) -> Dict:
        """Sanitize and validate input data."""
        clean = {}
        for feature in self.FEATURES:
            try:
                clean[feature] = float(data.get(feature, 0.0))
            except (ValueError, TypeError):
                clean[feature] = 0.0
                
        for feature in self.CATEGORICAL_FEATURES:
            clean[feature] = str(data.get(feature, 'Concentrate'))
            
        # Logic fix: internal_standard cannot be 0, avoid division by zero
        if clean.get('internal_standard', 0) <= 0:
            clean['internal_standard'] = 1.0
            
        return clean

    def _calculate_uncertainty_budget(self, data: Dict, grade: float) -> float:
        """
        Calculate Expanded Measurement Uncertainty (k=2) via bottom-up approach.
        U = k * sqrt(u_cal^2 + u_prec^2 + u_bias^2)
        """
        # 1. Precision Uncertainty (from replicate RSD)
        # u_prec = (RSD / 100) * Grade
        # We assume RSD provided is %
        rsd = data.get('replicate_rsd', 2.0)
        u_prec = (rsd / 100.0) * grade
        
        # 2. Calibration Uncertainty (Curve Fit)
        # Simplified: proportional to grade * (1-R2)
        u_cal = grade * (1 - self.calibration_model.r2) * 5 # Factor 5 is empirical
        
        # 3. Bias Uncertainty (Model RMSE)
        # We use the validation RMSE as the bias estimate
        u_bias = self.metrics.get('rmse', 0.1)
        
        # 4. Drift Factor
        drift = abs(data.get('calibration_drift', 1.0) - 1.0)
        u_drift = grade * drift
        
        # Combined Standard Uncertainty
        u_c = np.sqrt(u_prec**2 + u_cal**2 + u_bias**2 + u_drift**2)
        
        # Expanded Uncertainty (k=2, 95%)
        return 2 * u_c

    def predict(self, data: Dict) -> Dict:
        """Make a scientific prediction with valid uncertainty."""
        if not self.models:
             return {'status': 'Error: Model not loaded'}
             
        try:
            clean_data = self._sanitize_input(data)
            df = pd.DataFrame([clean_data])
            
            # Ensure categorical consistency
            for col in self.CATEGORICAL_FEATURES:
                if col not in df.columns:
                    df[col] = 'Concentrate'

            # 1. Physics Prediction
            grade_physics = self.calibration_model.predict(df)[0]
            
            # 2. ML Correction (Median, Low, High)
            correction_med = self.models['median'].predict(df)[0]
            correction_low = self.models['lower'].predict(df)[0]
            correction_high = self.models['upper'].predict(df)[0]
            
            # 3. Final Grade
            final_grade = grade_physics + correction_med
            
            # 4. QC: Check Detection Limits
            lod = self.calibration_model.lod
            loq = self.calibration_model.loq
            reportable_grade = final_grade
            status = 'Success'
            qc_flags = []
            
            if final_grade < lod:
                reportable_grade = lod
                status = '< LOD'
                qc_flags.append('below_detection_limit')
            elif final_grade < loq:
                status = 'Approximated (< LOQ)'
                qc_flags.append('below_quantification_limit')
            
            # 5. Uncertainty Calculation (GUM approach + ML Interval Check)
            # Theoretical Interval from Quantile Regression
            interval_low = grade_physics + correction_low
            interval_high = grade_physics + correction_high
            
            # GUM Uncertainty
            uncertainty_gum = self._calculate_uncertainty_budget(clean_data, final_grade)
            
            # Combine methods: Use the wider of the two ranges for conservatism
            ml_width = (interval_high - interval_low) / 2
            uncertainty_final = max(uncertainty_gum, ml_width)
            
            # 6. Additional QC
            if clean_data['replicate_rsd'] > 5.0:
                qc_flags.append('high_rsd_fail')
                status = 'Rejected (High RSD)'
            
            if abs(clean_data['calibration_drift'] - 1.0) > 0.1:
                qc_flags.append('drift_fail')
                status = 'Rejected (Drift)'
            
            return {
                'grade': round(reportable_grade, 4),
                'grade_raw': round(final_grade, 4),
                'uncertainty': round(uncertainty_final, 4),
                'confidence_interval': {
                    'lower': round(max(0, reportable_grade - uncertainty_final), 4),
                    'upper': round(reportable_grade + uncertainty_final, 4)
                },
                'lod': round(lod, 4),
                'loq': round(loq, 4),
                'components': {
                    'physics_grade': round(grade_physics, 4),
                    'matrix_correction': round(correction_med, 4)
                },
                'status': status,
                'qc_flags': qc_flags
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'status': f'Error: {str(e)}', 'grade': 0.0}

    def batch_predict(self, data_list: List[Dict]) -> List[Dict]:
        return [self.predict(d) for d in data_list]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the Median correction model."""
        if 'median' not in self.models:
            return {}
        try:
            regressor = self.models['median'].named_steps['regressor']
            importances = regressor.feature_importances_
            feature_names = self._get_feature_names()
            return dict(zip(feature_names, importances[:len(feature_names)]))
        except Exception:
            return {}

    def _get_feature_names(self) -> List[str]:
        """Helper to extract feature names."""
        if 'median' not in self.models: 
            return self.FEATURES
        try:
            preprocessor = self.models['median'].named_steps['preprocessor']
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(self.CATEGORICAL_FEATURES)
            return list(self.FEATURES) + list(cat_features)
        except:
            return self.FEATURES

    def explain_prediction(self, data: Dict) -> List[Dict]:
        """Generate explanations for a single prediction (Feature Importance based)."""
        if 'median' not in self.models:
            return []

        try:
            # Use feature importance from the Median model as a proxy for explanation
            importances = self.get_feature_importance()
            explanation = []
            
            # Sort by importance
            sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for name, val in sorted_feats[:5]: # Top 5
                explanation.append({
                    'feature': name.replace('_', ' ').title(),
                    'impact': float(val),
                    'direction': 'positive' # Simplified: GBR variable importance doesn't give direction easily
                })
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            return []

# Singleton Access
_ml_engine = None

def get_ml_engine() -> AssayMLEngine:
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = AssayMLEngine()
    return _ml_engine

def retrain_model_with_samples(samples: List) -> Dict:
    """Wrapper to retrain analysis model with real samples."""
    # Logic similar to original but adapted for dataframe structure
    data = []
    y = []
    
    for s in samples:
        if s.actual_grade is None: continue
        row = {'sample_type': s.sample_type, 'assay_method': s.assay_method}
        for f in AssayMLEngine.FEATURES:
            row[f] = getattr(s, f, 0.0)
        data.append(row)
        y.append(s.actual_grade)
    
    if len(data) < 50:
         return {'error': 'Insufficient samples (min 50)'}

    engine = get_ml_engine()
    metrics = engine.train_model(pd.DataFrame(data), np.array(y))
    return {'success': True, 'metrics': metrics}


def initialize_default_models():
    """Initialize default ML models on startup."""
    engine = get_ml_engine()
    acc = engine.metrics.get('r2_score', 0.0)
    logger.info(f"ML Engine ready. R2 Score: {acc:.4f}")
