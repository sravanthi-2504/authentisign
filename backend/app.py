"""
Flask Backend API for Signature Verification System
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
from datetime import datetime, timedelta
from functools import wraps
import os
import numpy as np
import cv2
from pathlib import Path
import uuid
import json

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('verification_history', exist_ok=True)

# In-memory user database
users_db = {
    'bsonakshi@gmail.com': {
        'password': generate_password_hash('password123'),
        'name': 'Sonakshi Bose',
        'created_at': datetime.now().isoformat()
    }
}

# Verification history storage
history_db = {}

# Load ML model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, '..', 'model', 'signature_model_final.h5')
)


model = None

# ✅ FIX: Custom distance function (MUST match training script)
def euclidean_distance(vectors):
    """Custom Euclidean distance for Lambda layer"""
    v1, v2 = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1, keepdims=True))

safe_mode=False
def load_model():
    """Load the trained model"""
    global model
    try:
        # Try .keras format first (recommended)
        keras_path = MODEL_PATH.replace('.h5', '.keras')

        print("Attempting to load model...")

        if os.path.exists(keras_path):
            print(f"Loading from: {keras_path}")
            # ✅ ADD safe_mode=False
            model = keras.models.load_model(keras_path, safe_mode=False)
            print("✓ Model loaded successfully (.keras format)")
        elif os.path.exists(MODEL_PATH):
            print(f"Loading from: {MODEL_PATH}")
            # ✅ ADD safe_mode=False here too
            model = keras.models.load_model(MODEL_PATH, safe_mode=False)
            print("✓ Model loaded successfully (.h5 format)")
        else:
            print(f"❌ Model file NOT found")
            print(f"   Checked: {keras_path}")
            print(f"   Checked: {MODEL_PATH}")
            return

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

# Load on startup
load_model()


def preprocess_signature(image_path, target_size=(128, 128)):  # ✅ FIX: Changed to 128x128
    """Preprocess signature image for model prediction"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # ✅ FIX: Resize to match training (128x128)
    img = cv2.resize(img, target_size)

    # Normalize to 0-1
    img = img.astype('float32') / 255.0

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)

    return img


def token_required(f):
    """Decorator to verify JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['email']
        except:
            return jsonify({'error': 'Token is invalid'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')
    name = data.get('name', email.split('@')[0])

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    if email in users_db:
        return jsonify({'error': 'User already exists'}), 409

    users_db[email] = {
        'password': generate_password_hash(password),
        'name': name,
        'created_at': datetime.now().isoformat()
    }

    # Initialize history for user
    history_db[email] = []

    return jsonify({
        'message': 'User registered successfully',
        'email': email,
        'name': name
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user and return JWT token"""
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    user = users_db.get(email)

    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    # Generate JWT token
    token = jwt.encode({
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        'token': token,
        'user': {
            'email': email,
            'name': user['name']
        }
    }), 200


@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token(current_user):
    """Verify if token is valid"""
    user = users_db.get(current_user)
    return jsonify({
        'user': {
            'email': current_user,
            'name': user['name']
        }
    }), 200


@app.route('/api/verify-signature', methods=['POST'])
@token_required
def verify_signature(current_user):
    """Verify signature using ML model"""

    if not model:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 503

    if 'original' not in request.files or 'test' not in request.files:
        return jsonify({'error': 'Both original and test signatures are required'}), 400

    original_file = request.files['original']
    test_file = request.files['test']

    if original_file.filename == '' or test_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save uploaded files
        original_filename = secure_filename(f"{uuid.uuid4()}_{original_file.filename}")
        test_filename = secure_filename(f"{uuid.uuid4()}_{test_file.filename}")

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)

        original_file.save(original_path)
        test_file.save(test_path)

        # ✅ FIX: Preprocess images (now correctly 128x128)
        img1 = preprocess_signature(original_path)
        img2 = preprocess_signature(test_path)

        # Add batch dimension
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        # Predict
        prediction = model.predict([img1, img2], verbose=0)[0][0]

        # Calculate results
        genuine_probability = float(prediction)
        forged_probability = float(1 - prediction)
        is_genuine = bool(prediction > 0.5)
        confidence = float(max(prediction, 1 - prediction) * 100)

        result = {
            'status': 'GENUINE' if is_genuine else 'FORGED',
            'confidence': round(confidence, 2),
            'genuine_probability': round(genuine_probability * 100, 2),
            'forged_probability': round(forged_probability * 100, 2),
            'timestamp': datetime.now().isoformat(),
            'filename': f"signature_comparison_{uuid.uuid4().hex[:8]}.jpg"
        }

        # Store in history
        if current_user not in history_db:
            history_db[current_user] = []

        history_entry = {
            'id': str(uuid.uuid4()),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'filename': result['filename'],
            'status': result['status'],
            'confidence': result['confidence'],
            'genuine_probability': result['genuine_probability'],
            'forged_probability': result['forged_probability']
        }

        history_db[current_user].insert(0, history_entry)

        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
@token_required
def get_history(current_user):
    """Get verification history for current user"""
    user_history = history_db.get(current_user, [])

    return jsonify({
        'history': user_history,
        'total': len(user_history)
    }), 200


@app.route('/api/history/<history_id>', methods=['DELETE'])
@token_required
def delete_history(current_user, history_id):
    """Delete a history entry"""
    if current_user not in history_db:
        return jsonify({'error': 'No history found'}), 404

    user_history = history_db[current_user]
    history_db[current_user] = [h for h in user_history if h['id'] != history_id]

    return jsonify({'message': 'History entry deleted'}), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("="*60)
    print("  AI SIGNATURE VERIFICATION BACKEND")
    print("="*60)
    print(f"✓ Server starting on http://localhost:5000")
    print(f"✓ Model status: {'Loaded' if model else 'Not loaded'}")
    print(f"✓ CORS enabled for: http://localhost:3000, http://localhost:5173")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)