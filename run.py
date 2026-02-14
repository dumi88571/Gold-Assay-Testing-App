"""
Gold Assay Analyzer - Professional Edition
===========================================

Entry point for the Flask application.

Usage:
    python run.py              # Run development server
    python run.py --init-db    # Initialize database only
    python run.py --retrain    # Retrain ML model

Environment Variables:
    FLASK_ENV=development|production
    SECRET_KEY=your-secret-key
    DATABASE_URL=sqlite:///... or postgresql://...
"""

import os
import sys
import argparse

from app import create_app, db
from app.models import User, Sample, Batch
from app.services.ml_engine import get_ml_engine, retrain_model_with_samples


def init_database():
    """Initialize database with tables and default data."""
    app = create_app()
    with app.app_context():
        db.create_all()
        print("Database tables created.")
        
        # Check if admin user exists
        if not User.query.filter_by(role='admin').first():
            print("\nNo admin user found. Register the first user to become admin.")
        
        print("Database initialized successfully!")


def retrain_model():
    """Retrain ML model with validated samples."""
    app = create_app()
    with app.app_context():
        # Get samples with actual grades
        samples = Sample.query.filter(Sample.actual_grade.isnot(None)).all()
        
        if len(samples) < 50:
            print(f"Insufficient validated samples: {len(samples)} (need 50+)")
            return
        
        print(f"Retraining model with {len(samples)} validated samples...")
        result = retrain_model_with_samples(samples)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Model retrained successfully!")
            print(f"New accuracy: {result['metrics']['r2_score']:.4f}")


def run_tests():
    """Run basic application tests."""
    app = create_app()
    with app.app_context():
        # Test ML engine
        engine = get_ml_engine()
        print(f"ML Engine Status: OK (Accuracy: {engine.metrics.get('r2_score', 0.0):.4f})")
        
        # Test database
        try:
            count = Sample.query.count()
            print(f"Database Status: OK ({count} samples in database)")
        except Exception as e:
            print(f"Database Status: Error - {e}")


def main():
    parser = argparse.ArgumentParser(description='Gold Assay Analyzer')
    parser.add_argument('--init-db', action='store_true', 
                       help='Initialize database and exit')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain ML model with validated samples')
    parser.add_argument('--test', action='store_true',
                       help='Run basic tests')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.init_db:
        init_database()
        return
    
    if args.retrain:
        retrain_model()
        return
    
    if args.test:
        run_tests()
        return
    
    # Run development server
    app = create_app()
    print(f"""
╔════════════════════════════════════════════════════════════╗
║          Gold Assay Analyzer - Professional Edition        ║
║                                                            ║
║  Starting server at http://{args.host}:{args.port:<5}               ║
║  Debug mode: {'ON' if args.debug else 'OFF':<3}                                  ║
╚════════════════════════════════════════════════════════════╝
    """)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
