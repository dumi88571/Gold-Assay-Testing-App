"""
Gold Assay Analyzer - Professional Edition
===========================================

A production-ready Flask application for gold assay data analysis.
Features robust ML prediction, statistical analysis, calibration management,
and comprehensive laboratory reporting.

Author: AI Mining Solutions
Version: 10.0.0 (Professional Edition)
"""

import os
import logging
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()


def create_app(config_class=Config):
    """Application factory pattern for creating Flask app."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    app.config.from_object(config_class)
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    # Register blueprints
    from app.routes.main import bp as main_bp
    from app.routes.analysis import bp as analysis_bp
    from app.routes.api import bp as api_bp
    from app.routes.tools import bp as tools_bp
    from app.routes.auth import bp as auth_bp
    from app.routes.calibration import bp as calibration_bp
    from app.routes.statistics import bp as statistics_bp
    from app.routes.batch import bp as batch_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp, url_prefix='/analyze')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(tools_bp, url_prefix='/tools')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(calibration_bp, url_prefix='/calibration')
    app.register_blueprint(statistics_bp, url_prefix='/stats')
    app.register_blueprint(batch_bp, url_prefix='/batch')
    
    # Create database tables
    with app.app_context():
        db.create_all()
        
        # Initialize default data
        from app.services.ml_engine import initialize_default_models
        initialize_default_models()
        
        from app.models import CalibrationStandard
        CalibrationStandard.initialize_defaults()
    
    # Setup logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = logging.FileHandler(f'logs/assay_{datetime.now().strftime("%Y%m")}.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Gold Assay Analyzer startup')
    
    return app


from app import models
