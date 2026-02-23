"""
Fixed Signature Verification using Euclidean Distance
(NOT probability - use distance as metric)
"""

import numpy as np
import tensorflow as tf
import cv2
import os
from pathlib import Path

class SignatureVerifierFixed:
    """
    Load trained Siamese model and verify signatures
    Uses EUCLIDEAN DISTANCE (not probability)
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "signature_model_final.keras"
            )

        model_path = os.path.abspath(model_path)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded")

        self.target_size = (128, 128)
        # ✅ THRESHOLD = 0.4-0.6 (euclidean distance)
        #    < 0.5 = GENUINE, >= 0.5 = FORGED
        self.THRESHOLD = 0.5

    def preprocess(self, image_path):
        """Exact same preprocessing as training"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")

        img = cv2.resize(img, self.target_size)

        # Adaptive threshold
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=-1)

    def verify(self, original_path, test_path):
        """Verify signature pair"""
        try:
            img1 = np.expand_dims(self.preprocess(original_path), 0)
            img2 = np.expand_dims(self.preprocess(test_path), 0)

            # Get euclidean distance
            distance = float(self.model.predict([img1, img2], verbose=0)[0][0])

            # Decision
            is_genuine = distance < self.THRESHOLD

            # Confidence based on distance from threshold
            if is_genuine:
                confidence = max(0, (self.THRESHOLD - distance) / self.THRESHOLD * 100)
            else:
                confidence = max(0, (distance - self.THRESHOLD) / (2 - self.THRESHOLD) * 100)

            return {
                "status": "GENUINE" if is_genuine else "FORGED",
                "confidence": round(min(confidence, 99.99), 2),
                "distance": round(distance, 4),
                "threshold": self.THRESHOLD
            }

        except Exception as e:
            raise Exception(f"Verification failed: {str(e)}")


if __name__ == "__main__":
    verifier = SignatureVerifierFixed()

    # Test
    result = verifier.verify("sig1.png", "sig2.png")
    print(f"Status: {result['status']}")
    print(f"Distance: {result['distance']}")
    print(f"Confidence: {result['confidence']}%")