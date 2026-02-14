"""Authentication routes."""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime

from app import db, login_manager
from app.models import User, AuditLog
from app.schemas import UserCreate, UserLogin

bp = Blueprint('auth', __name__)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Account is deactivated', 'danger')
                return redirect(url_for('auth.login'))
            
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            AuditLog.log_action(
                action='login',
                entity_type='User',
                entity_id=user.id,
                notes=f'User {username} logged in'
            )
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('main.index'))
        
        flash('Invalid username or password', 'danger')
    
    return render_template('login.html')


@bp.route('/logout')
@login_required
def logout():
    """User logout."""
    AuditLog.log_action(
        action='logout',
        entity_type='User',
        entity_id=current_user.id,
        notes=f'User {current_user.username} logged out'
    )
    
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))


@bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration (admin only in production)."""
    # Check if any users exist - first user becomes admin
    existing_users = User.query.count()
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        
        # Check for existing user
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('auth.register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('auth.register'))
        
        # Create user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            role='admin' if existing_users == 0 else 'analyst'
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        AuditLog.log_action(
            action='create',
            entity_type='User',
            entity_id=user.id,
            notes=f'New user registered: {username}'
        )
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html', first_user=(existing_users == 0))


@bp.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('profile.html', user=current_user)


@bp.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    """Update user profile."""
    user = current_user
    
    old_values = {
        'email': user.email,
        'full_name': user.full_name
    }
    
    user.email = request.form.get('email', user.email)
    user.full_name = request.form.get('full_name', user.full_name)
    
    # Password change
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    
    if current_password and new_password:
        if user.check_password(current_password):
            user.set_password(new_password)
            flash('Password updated successfully', 'success')
        else:
            flash('Current password is incorrect', 'danger')
            return redirect(url_for('auth.profile'))
    
    db.session.commit()
    
    AuditLog.log_action(
        action='update',
        entity_type='User',
        entity_id=user.id,
        old_values=old_values,
        new_values={'email': user.email, 'full_name': user.full_name},
        notes='Profile updated'
    )
    
    flash('Profile updated', 'success')
    return redirect(url_for('auth.profile'))


@bp.route('/users')
@login_required
def list_users():
    """List all users (admin only)."""
    if not current_user.has_role('admin'):
        flash('Admin access required', 'danger')
        return redirect(url_for('main.index'))
    
    users = User.query.all()
    return render_template('users.html', users=users)


@bp.route('/users/<int:user_id>/toggle', methods=['POST'])
@login_required
def toggle_user(user_id):
    """Activate/deactivate user (admin only)."""
    if not current_user.has_role('admin'):
        flash('Admin access required', 'danger')
        return redirect(url_for('main.index'))
    
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('Cannot deactivate yourself', 'danger')
        return redirect(url_for('auth.list_users'))
    
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    AuditLog.log_action(
        action='update',
        entity_type='User',
        entity_id=user.id,
        notes=f'User {status} by admin'
    )
    
    flash(f'User {user.username} {status}', 'success')
    return redirect(url_for('auth.list_users'))
