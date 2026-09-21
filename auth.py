"""
Authentication routes and utilities for ChemoGuard.
"""

from flask import Blueprint, request, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from database import db, User
from datetime import datetime

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint."""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or not all(k in data for k in ['username', 'email', 'password']):
            return jsonify({'error': 'Username, email, and password are required'}), 400
        
        username = data.get('username').strip()
        email = data.get('email').strip().lower()
        password = data.get('password')
        full_name = data.get('full_name', '').strip()
        
        # Validate password strength
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 409
        
        # Create new user
        user = User(
            username=username,
            email=email,
            full_name=full_name
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Auto-login after registration
        login_user(user, remember=True)
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name
            }
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint."""
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['username', 'password']):
            return jsonify({'error': 'Username and password are required'}), 400
        
        username = data.get('username').strip()
        password = data.get('password')
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username) | (User.email == username.lower())
        ).first()
        
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Login user
        remember = data.get('remember', False)
        login_user(user, remember=remember)
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """User logout endpoint."""
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200


@auth_bp.route('/profile', methods=['GET'])
@login_required
def get_profile():
    """Get current user profile."""
    try:
        user = current_user
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'created_at': user.created_at.isoformat() if user.created_at else None
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile."""
    try:
        data = request.get_json()
        user = current_user
        
        if 'full_name' in data:
            user.full_name = data['full_name'].strip()
        
        if 'email' in data:
            new_email = data['email'].strip().lower()
            # Check if email is already taken by another user
            existing_user = User.query.filter_by(email=new_email).first()
            if existing_user and existing_user.id != user.id:
                return jsonify({'error': 'Email already registered'}), 409
            user.email = new_email
        
        if 'password' in data and data.get('old_password'):
            # Verify old password
            if not user.check_password(data['old_password']):
                return jsonify({'error': 'Current password is incorrect'}), 401
            # Set new password
            if len(data['password']) < 6:
                return jsonify({'error': 'Password must be at least 6 characters long'}), 400
            user.set_password(data['password'])
        
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name
            }
        }), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/check_auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated."""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email
            }
        }), 200
    else:
        return jsonify({'authenticated': False}), 200

