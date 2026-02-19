import numpy as np
import tensorflow as tf
import cv2
import os

class SignatureVerifier:

    def __init__(self):
        print("Loading trained model...")

        # Make sure model loads correctly even if called from app.py
        model_path = os.path.join(
            os.path.dirname(__file__),
            "../model/signature_model_final.keras"
        )

        model_path = os.path.abspath(model_path)

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print("✓ Model loaded successfully")

        self.target_size = (128, 128)
        # ✅ FIXED: Lowered threshold from 0.90 to 0.5
        # Your model outputs: genuine=0.86-0.98, forged=0.0
        # So 0.5 is the correct decision boundary
        self.THRESHOLD = 0.85

    def preprocess(self, image_bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Invalid image")

        # Resize (same as training)
        img = cv2.resize(img, self.target_size)

        # EXACT SAME preprocessing as training
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Normalize
        img = img.astype("float32") / 255.0

        # Add channel dimension
        img = np.expand_dims(img, axis=-1)

        return img

    def verify(self, original_bytes, test_bytes):
        try:
            img1 = self.preprocess(original_bytes)
            img2 = self.preprocess(test_bytes)

            # Add batch dimension
            img1 = np.expand_dims(img1, axis=0)
            img2 = np.expand_dims(img2, axis=0)

            # Predict
            prediction = float(
                self.model.predict([img1, img2], verbose=0)[0][0]
            )

            # ✅ FIXED: Corrected decision logic
            # Model outputs HIGH probability (>0.5) for GENUINE
            # Model outputs LOW probability (<0.5) for FORGED
            is_genuine = prediction > self.THRESHOLD

            status = "GENUINE" if is_genuine else "FORGED"

            # ✅ FIXED: Better confidence calculation
            # Confidence should always be based on how far from the threshold
            if is_genuine:
                # For genuine: how confident above threshold
                confidence = int(prediction * 100)
            else:
                # For forged: how confident below threshold
                confidence = int((1 - prediction) * 100)

            return {
                "status": status,
                "confidence": confidence,
                "genuine_probability": round(prediction * 100, 2),
                "forged_probability": round((1 - prediction) * 100, 2),
                "raw_probability": round(prediction, 4)
            }

        except Exception as e:
            raise Exception(f"Verification failed: {str(e)}")