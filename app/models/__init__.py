"""Database models for Gold Assay Analyzer."""

import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(UserMixin, db.Model):
    """User account model."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    full_name = db.Column(db.String(128))
    role = db.Column(db.String(20), default='analyst')  # admin, analyst, technician, viewer
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    samples = db.relationship('Sample', backref='analyst', lazy='dynamic')
    calibrations = db.relationship('CalibrationRecord', backref='performed_by', lazy='dynamic')
    audit_logs = db.relationship('AuditLog', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def has_role(self, role):
        return self.role == role or self.role == 'admin'
    
    def __repr__(self):
        return f'<User {self.username}>'


class Sample(db.Model):
    """Sample analysis record model."""
    __tablename__ = 'samples'
    
    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.String(20), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Sample metadata
    sample_type = db.Column(db.String(50), nullable=False)  # Concentrate, Tailings, Solution, Feed
    assay_method = db.Column(db.String(50), nullable=False)  # Fire Assay, Acid Digest, Cyanide
    client_name = db.Column(db.String(128))
    batch_id = db.Column(db.String(20), db.ForeignKey('batches.batch_id'))
    
    # Analytical parameters (17 features)
    absorption_242nm = db.Column(db.Float)
    absorption_267nm = db.Column(db.Float)
    emission_intensity = db.Column(db.Float)
    solution_ph = db.Column(db.Float)
    temperature_c = db.Column(db.Float)
    ionic_strength = db.Column(db.Float)
    dissolved_oxygen = db.Column(db.Float)
    iron_ppm = db.Column(db.Float)
    copper_ppm = db.Column(db.Float)
    silver_ppm = db.Column(db.Float)
    sulfur_content = db.Column(db.Float)
    dilution_factor = db.Column(db.Float)
    measurement_time = db.Column(db.Float)
    calibration_drift = db.Column(db.Float)
    replicate_rsd = db.Column(db.Float)
    blank_intensity = db.Column(db.Float)
    internal_standard = db.Column(db.Float)
    
    # Results
    predicted_grade = db.Column(db.Float)
    actual_grade = db.Column(db.Float)  # For training/validation
    model_confidence = db.Column(db.Float)
    uncertainty = db.Column(db.Float)
    status = db.Column(db.String(20), default='completed')  # pending, completed, failed, rejected
    
    # Quality flags
    qc_passed = db.Column(db.Boolean, default=True)
    qc_flags = db.Column(db.JSON)
    
    # Foreign keys
    analyst_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    calibration_id = db.Column(db.Integer, db.ForeignKey('calibration_records.id'))
    
    def generate_sample_id(self):
        """Generate unique sample ID."""
        self.sample_id = f"S-{datetime.now().strftime('%y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    
    def to_dict(self):
        """Convert sample to dictionary."""
        return {
            'id': self.id,
            'sample_id': self.sample_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'sample_type': self.sample_type,
            'assay_method': self.assay_method,
            'predicted_grade': self.predicted_grade,
            'actual_grade': self.actual_grade,
            'model_confidence': self.model_confidence,
            'status': self.status,
            'qc_passed': self.qc_passed,
            'features': {
                'absorption_242nm': self.absorption_242nm,
                'absorption_267nm': self.absorption_267nm,
                'emission_intensity': self.emission_intensity,
                'solution_ph': self.solution_ph,
                'temperature_c': self.temperature_c,
                'ionic_strength': self.ionic_strength,
                'dissolved_oxygen': self.dissolved_oxygen,
                'iron_ppm': self.iron_ppm,
                'copper_ppm': self.copper_ppm,
                'silver_ppm': self.silver_ppm,
                'sulfur_content': self.sulfur_content,
                'dilution_factor': self.dilution_factor,
                'measurement_time': self.measurement_time,
                'calibration_drift': self.calibration_drift,
                'replicate_rsd': self.replicate_rsd,
                'blank_intensity': self.blank_intensity,
                'internal_standard': self.internal_standard
            }
        }
    
    def __repr__(self):
        return f'<Sample {self.sample_id}>'


class Batch(db.Model):
    """Batch processing model for multiple samples."""
    __tablename__ = 'batches'
    
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(20), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    
    # Batch metadata
    batch_name = db.Column(db.String(128))
    sample_count = db.Column(db.Integer, default=0)
    client_name = db.Column(db.String(128))
    
    # Source file info
    source_filename = db.Column(db.String(256))
    file_hash = db.Column(db.String(64))
    
    # Results summary
    avg_grade = db.Column(db.Float)
    min_grade = db.Column(db.Float)
    max_grade = db.Column(db.Float)
    std_dev = db.Column(db.Float)
    
    # Relationships
    samples = db.relationship('Sample', backref='batch', lazy='dynamic')
    
    def generate_batch_id(self):
        """Generate unique batch ID."""
        self.batch_id = f"B-{datetime.now().strftime('%y%m%d')}-{uuid.uuid4().hex[:4].upper()}"


class CalibrationRecord(db.Model):
    """Instrument calibration record."""
    __tablename__ = 'calibration_records'
    
    id = db.Column(db.Integer, primary_key=True)
    calibration_date = db.Column(db.DateTime, default=datetime.utcnow)
    instrument_id = db.Column(db.String(50))
    analyst_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Calibration details
    calibration_type = db.Column(db.String(50))  # daily, weekly, monthly, maintenance
    standard_used = db.Column(db.String(50))
    drift_check = db.Column(db.Float)
    slope = db.Column(db.Float)
    intercept = db.Column(db.Float)
    r_squared = db.Column(db.Float)
    
    # Status
    passed = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    
    # Relationships
    samples = db.relationship('Sample', backref='calibration', lazy='dynamic')


class CalibrationStandard(db.Model):
    """Calibration standard reference materials."""
    __tablename__ = 'calibration_standards'
    
    id = db.Column(db.Integer, primary_key=True)
    standard_name = db.Column(db.String(50), unique=True)
    certificate_id = db.Column(db.String(50))
    gold_grade = db.Column(db.Float)
    uncertainty = db.Column(db.Float)
    expiry_date = db.Column(db.Date)
    is_active = db.Column(db.Boolean, default=True)
    
    @classmethod
    def initialize_defaults(cls):
        """Initialize default calibration standards."""
        defaults = [
            ('Au-Standard-Low', 0.5, 0.02),
            ('Au-Standard-Med', 2.5, 0.05),
            ('Au-Standard-High', 10.0, 0.1),
            ('Au-Standard-Very-High', 50.0, 0.5)
        ]
        for name, grade, unc in defaults:
            if not cls.query.filter_by(standard_name=name).first():
                db.session.add(cls(
                    standard_name=name,
                    certificate_id=f"CERT-{name}",
                    gold_grade=grade,
                    uncertainty=unc
                ))
        db.session.commit()


class AuditLog(db.Model):
    """Audit trail for compliance."""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    action = db.Column(db.String(50), nullable=False)  # create, update, delete, export, login, etc.
    entity_type = db.Column(db.String(50))  # Sample, Batch, Calibration, User
    entity_id = db.Column(db.String(50))
    
    # User info
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    username = db.Column(db.String(64))
    ip_address = db.Column(db.String(45))
    
    # Change details
    old_values = db.Column(db.JSON)
    new_values = db.Column(db.JSON)
    notes = db.Column(db.Text)
    
    @staticmethod
    def log_action(action, entity_type=None, entity_id=None, 
                   old_values=None, new_values=None, notes=None):
        """Log an action to the audit trail."""
        from flask_login import current_user
        from flask import request
        
        log = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id else None,
            user_id=current_user.id if current_user.is_authenticated else None,
            username=current_user.username if current_user.is_authenticated else 'anonymous',
            ip_address=request.remote_addr if request else None,
            old_values=old_values,
            new_values=new_values,
            notes=notes
        )
        db.session.add(log)
        db.session.commit()


class StatisticalSummary(db.Model):
    """Pre-computed statistical summaries for reporting."""
    __tablename__ = 'statistical_summaries'
    
    id = db.Column(db.Integer, primary_key=True)
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)
    period_start = db.Column(db.DateTime)
    period_end = db.Column(db.DateTime)
    
    # Summary stats
    sample_count = db.Column(db.Integer)
    avg_grade = db.Column(db.Float)
    median_grade = db.Column(db.Float)
    std_dev = db.Column(db.Float)
    min_grade = db.Column(db.Float)
    max_grade = db.Column(db.Float)
    
    # Additional metrics
    qc_pass_rate = db.Column(db.Float)
    model_accuracy = db.Column(db.Float)
    
    # Filter criteria used
    filters = db.Column(db.JSON)
