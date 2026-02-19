"""
Flask Backend - COMPLETELY FIXED
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
from datetime import datetime, timedelta
from functools import wraps
import os, numpy as np, cv2, uuid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AbsoluteLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)
    def get_config(self):
        return super().get_config()


app = Flask(__name__)
app.config["SECRET_KEY"]         = "your-secret-key-change-in-production"
app.config["UPLOAD_FOLDER"]      = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

CORS(app, resources={r"/*": {
    "origins":       ["http://localhost:3000", "http://localhost:5173", "http://localhost:3001"],
    "methods":       ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}})

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("verification_history",      exist_ok=True)

users_db = {
    "bsonakshi@gmail.com": {
        "password":   generate_password_hash("password123"),
        "name":       "Sonakshi Bose",
        "created_at": datetime.now().isoformat(),
    },
    "csravanthi2006@gmail.com": {
        "password":   generate_password_hash("password123"),
        "name":       "C Sravanthi",
        "created_at": datetime.now().isoformat(),
    }
}
history_db = {}

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))

model = None


def load_model():
    global model
    custom = {"AbsoluteLayer": AbsoluteLayer}

    print("\nLoading model...")

    keras_path = os.path.join(MODEL_DIR, "signature_model_final.keras")
    if os.path.exists(keras_path):
        print(f"  Found: {keras_path}")
        try:
            model = keras.models.load_model(keras_path, custom_objects=custom, safe_mode=False)
            print("  ✓ Loaded .keras")
            return
        except Exception as e:
            print(f"  ✗ Error: {e}")

    h5_path = os.path.join(MODEL_DIR, "signature_model_final.h5")
    if os.path.exists(h5_path):
        print(f"  Found: {h5_path}")
        try:
            model = keras.models.load_model(h5_path, custom_objects=custom, safe_mode=False)
            print("  ✓ Loaded .h5")
            return
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("  ❌ No model found - train first!")


load_model()


# ✅ EXACT COPY from train_model.py line 65-77
def preprocess_signature(path):
    """EXACT COPY from train_model.py CEDARPairGenerator.preprocess()"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {path}")

    img = cv2.resize(img, (128, 128))

    # Apply thresholding for better contrast
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = data["email"]
        except Exception:
            return jsonify({"error": "Token is invalid"}), 401
        return f(current_user, *args, **kwargs)
    return decorated


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    email, password = data.get("email"), data.get("password")
    name = data.get("name", (email or "").split("@")[0])
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    if email in users_db:
        return jsonify({"error": "User exists"}), 409
    users_db[email] = {"password": generate_password_hash(password),
                       "name": name, "created_at": datetime.now().isoformat()}
    history_db[email] = []
    return jsonify({"message": "Registered", "email": email, "name": name}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email, password = data.get("email"), data.get("password")
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    user = users_db.get(email)
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401
    token = jwt.encode(
        {"email": email, "exp": datetime.utcnow() + timedelta(hours=24)},
        app.config["SECRET_KEY"], algorithm="HS256")
    return jsonify({"token": token, "user": {"email": email, "name": user["name"]}}), 200


@app.route("/api/auth/verify", methods=["GET"])
@token_required
def verify_token(current_user):
    user = users_db.get(current_user)
    return jsonify({"user": {"email": current_user, "name": user["name"]}}), 200


@app.route("/api/verify-signature", methods=["POST"])
@token_required
def verify_signature(current_user):
    if not model:
        return jsonify({"error": "Model not loaded. Train first."}), 503

    if "original" not in request.files or "test" not in request.files:
        return jsonify({"error": "Both signatures required"}), 400

    orig_file = request.files["original"]
    test_file = request.files["test"]

    if not orig_file.filename or not test_file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save uploads
        oname = secure_filename(f"{uuid.uuid4()}_{orig_file.filename}")
        tname = secure_filename(f"{uuid.uuid4()}_{test_file.filename}")
        opath = os.path.join(app.config["UPLOAD_FOLDER"], oname)
        tpath = os.path.join(app.config["UPLOAD_FOLDER"], tname)
        orig_file.save(opath)
        test_file.save(tpath)

        # Preprocess
        img1 = np.expand_dims(preprocess_signature(opath), 0)
        img2 = np.expand_dims(preprocess_signature(tpath), 0)

        # DEBUG: Save preprocessed images to see what model receives
        debug_dir = "debug_preprocessing"
        os.makedirs(debug_dir, exist_ok=True)

        # Save original uploads
        import shutil
        shutil.copy(opath, os.path.join(debug_dir, "1_uploaded_original.png"))
        shutil.copy(tpath, os.path.join(debug_dir, "2_uploaded_test.png"))

        # Save preprocessed versions
        cv2.imwrite(os.path.join(debug_dir, "3_preprocessed_original.png"),
                    (img1[0,:,:,0] * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, "4_preprocessed_test.png"),
                    (img2[0,:,:,0] * 255).astype(np.uint8))

        print(f"\n[DEBUG] Saved preprocessed images to {debug_dir}/")
        print(f"[DEBUG] img1 shape: {img1.shape}, min: {img1.min():.3f}, max: {img1.max():.3f}")
        print(f"[DEBUG] img2 shape: {img2.shape}, min: {img2.min():.3f}, max: {img2.max():.3f}")

        # Predict - model outputs PROBABILITY
        raw_prob = float(model.predict([img1, img2], verbose=0)[0][0])

        print(f"[DEBUG] Raw probability: {raw_prob:.6f}")
        print(f"[DEBUG] Model says: {'GENUINE' if raw_prob > 0.5 else 'FORGED'}\n")

        # Decision with threshold
        THRESHOLD    = 0.5  # Standard threshold
        is_genuine   = raw_prob > THRESHOLD
        genuine_prob = raw_prob
        forged_prob  = 1.0 - raw_prob
        confidence   = max(genuine_prob, forged_prob) * 100

        result = {
            "status":               "GENUINE" if is_genuine else "FORGED",
            "confidence":           round(confidence, 2),
            "genuine_probability":  round(genuine_prob * 100, 2),
            "forged_probability":   round(forged_prob  * 100, 2),
            "raw_probability":      round(raw_prob, 4),
            "timestamp":            datetime.now().isoformat(),
            "filename":             f"comparison_{uuid.uuid4().hex[:8]}.jpg",
        }

        # Save to history
        history_db.setdefault(current_user, []).insert(0, {
            "id":   str(uuid.uuid4()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            **{k: result[k] for k in ("filename", "status", "confidence",
                                      "genuine_probability", "forged_probability")},
        })

        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Verification failed: {e}"}), 500


@app.route("/api/history", methods=["GET"])
@token_required
def get_history(current_user):
    h = history_db.get(current_user, [])
    return jsonify({"history": h, "total": len(h)}), 200


@app.route("/api/history/<hid>", methods=["DELETE"])
@token_required
def delete_history(current_user, hid):
    if current_user not in history_db:
        return jsonify({"error": "No history"}), 404
    history_db[current_user] = [x for x in history_db[current_user] if x["id"] != hid]
    return jsonify({"message": "Deleted"}), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    print("=" * 60)
    print("  AI SIGNATURE VERIFICATION BACKEND")
    print("=" * 60)
    print(f"  Server → http://localhost:5000")
    print(f"  Model  → {'Loaded ✓' if model else 'NOT loaded!'}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)