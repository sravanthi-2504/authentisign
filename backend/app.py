"""
Flask Backend - Production Ready
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
from datetime import datetime, timedelta
from functools import wraps
import os
import numpy as np
import cv2
import uuid
from model.train_model import L2Normalization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-change-in-production"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

CORS(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

users_db = {
    "bsonakshi@gmail.com": {
        "password": generate_password_hash("password123"),
        "name": "Sonakshi Bose",
    },
    "csravanthi2006@gmail.com": {
        "password": generate_password_hash("password123"),
        "name": "C Sravanthi",
    }
}

history_db = {}
embedding_model = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "signature_embedding_model.keras")
THRESHOLD = 0.5


def load_model():
    """Load embedding model"""
    global embedding_model

    print("Loading model...")

    if os.path.exists(MODEL_PATH):
        try:
            embedding_model = tf.keras.models.load_model(MODEL_PATH, compile=False,safe_mode=False, custom_objects={"L2Normalization": L2Normalization})
            print(f"✓ Model loaded\n")
            return True
        except Exception as e:
            print(f"✗ Error: {e}\n")

    print("❌ Model not found!\n")
    return False


load_model()


def preprocess(image_path):
    """Preprocess image"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    img = cv2.resize(img, (128, 128))

    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").split()[-1] \
            if "Authorization" in request.headers else None

        if not token:
            return jsonify({"error": "Token missing"}), 401

        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = data["email"]
        except:
            return jsonify({"error": "Invalid token"}), 401

        return f(current_user, *args, **kwargs)

    return decorated


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    name = data.get("name", email.split("@")[0] if email else "User")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if email in users_db:
        return jsonify({"error": "User exists"}), 409

    users_db[email] = {
        "password": generate_password_hash(password),
        "name": name,
    }
    history_db[email] = []

    return jsonify({"message": "Registered", "email": email}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user = users_db.get(email)
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode(
        {"email": email, "exp": datetime.utcnow() + timedelta(hours=24)},
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )

    return jsonify({
        "token": token,
        "user": {"email": email, "name": user["name"]}
    }), 200


@app.route("/api/auth/verify", methods=["GET"])
@token_required
def verify_token(current_user):
    user = users_db.get(current_user)
    return jsonify({"user": {"email": current_user, "name": user["name"]}}), 200


@app.route("/api/verify-signature", methods=["POST"])
@token_required
def verify_signature(current_user):
    """Verify signatures"""
    if not embedding_model:
        return jsonify({"error": "Model not loaded"}), 503

    if "original" not in request.files or "test" not in request.files:
        return jsonify({"error": "Both files required"}), 400

    orig_file = request.files["original"]
    test_file = request.files["test"]

    if not orig_file.filename or not test_file.filename:
        return jsonify({"error": "No files"}), 400

    try:
        # Save files
        orig_name = secure_filename(f"{uuid.uuid4()}_{orig_file.filename}")
        test_name = secure_filename(f"{uuid.uuid4()}_{test_file.filename}")
        orig_path = os.path.join(app.config["UPLOAD_FOLDER"], orig_name)
        test_path = os.path.join(app.config["UPLOAD_FOLDER"], test_name)

        orig_file.save(orig_path)
        test_file.save(test_path)

        # Preprocess
        img1 = preprocess(orig_path)
        img2 = preprocess(test_path)

        # Get embeddings
        embed1 = embedding_model.predict(np.expand_dims(img1, 0), verbose=0)[0]
        embed2 = embedding_model.predict(np.expand_dims(img2, 0), verbose=0)[0]

        # L2 distance
        distance = float(np.sqrt(np.sum((embed1 - embed2) ** 2)))

        # Decision
        is_genuine = distance < THRESHOLD

        # Confidence
        if is_genuine:
            confidence = (1.0 - distance / THRESHOLD) * 100
        else:
            confidence = ((distance - THRESHOLD) / (2.0 - THRESHOLD)) * 100

        result = {
            "status": "GENUINE" if is_genuine else "FORGED",
            "confidence": round(min(max(confidence, 0), 99.99), 2),
            "distance": round(distance, 4),
            "timestamp": datetime.now().isoformat(),
        }

        # Save to history
        history_db.setdefault(current_user, []).insert(0, {
            "id": str(uuid.uuid4()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": result["status"],
            "confidence": result["confidence"],
        })

        # Cleanup
        try:
            os.remove(orig_path)
            os.remove(test_path)
        except:
            pass

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
@token_required
def get_history(current_user):
    h = history_db.get(current_user, [])
    return jsonify({"history": h}), 200


@app.route("/api/history/<hid>", methods=["DELETE"])
@token_required
def delete_history(current_user, hid):
    if current_user not in history_db:
        return jsonify({"error": "No history"}), 404

    history_db[current_user] = [
        x for x in history_db[current_user] if x["id"] != hid
    ]

    return jsonify({"message": "Deleted"}), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": embedding_model is not None,
    }), 200


if __name__ == "__main__":
    print("="*70)
    print("  SIGNATURE VERIFICATION - BACKEND")
    print("="*70)
    print(f"  Model: {'✓ Loaded' if embedding_model else '✗ NOT LOADED'}\n")

    app.run(debug=True, host="0.0.0.0", port=5000)