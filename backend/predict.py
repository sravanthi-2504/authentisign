import numpy as np
import tensorflow as tf
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.train_model import L2Normalization
THRESHOLD = 0.40 # <-- Replace with your best threshold

class Verifier:

    def __init__(self):
        self.model = tf.keras.models.load_model(
            os.path.join(os.path.dirname(__file__), "..", "model", "signature_embedding_model.keras"),
            compile=False,
            safe_mode=False,
            custom_objects={"L2Normalization": L2Normalization}
        )

    def preprocess(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (128, 128))

        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    # ✅ THIS WAS MISSING — must match training exactly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        img = img.astype("float32") / 255.0
        return np.expand_dims(img, -1)

    def verify(self, p1, p2):
        i1 = np.expand_dims(self.preprocess(p1),0)
        i2 = np.expand_dims(self.preprocess(p2),0)

        e1 = self.model.predict(i1,verbose=0)[0]
        e2 = self.model.predict(i2,verbose=0)[0]

        d = np.sqrt(np.sum((e1-e2)**2))
        status = "GENUINE" if d < THRESHOLD else "FORGED"

        print("\nDistance:", round(d,4))
        print("Result:", status)

if __name__ == "__main__":
    Verifier().verify(sys.argv[1], sys.argv[2])