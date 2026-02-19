"""
Test your model on actual CEDAR dataset pairs
This will show you EXACTLY what probabilities your model outputs
"""

import os
import numpy as np
import cv2
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────
#  Custom Layer (if used in training)
# ─────────────────────────────────────────────────────────────
class AbsoluteLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────
#  Preprocessing (must match training)
# ─────────────────────────────────────────────────────────────
def preprocess(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)


# ─────────────────────────────────────────────────────────────
#  Main Test Function
# ─────────────────────────────────────────────────────────────
def test_model():

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    print("Loading model...")
    model = keras.models.load_model(
        "signature_model_final.keras",
        custom_objects={"AbsoluteLayer": AbsoluteLayer},
        safe_mode=False
    )
    print("✓ Model loaded\n")

    dataset = Path("../dataset/cedar_dataset")
    org_path  = dataset / "full_org"
    forg_path = dataset / "full_forg"

    print("=" * 70)
    print("TESTING ON CEDAR DATASET")
    print("=" * 70)

    # ─────────────────────────
    # 1️⃣ Genuine pairs
    # ─────────────────────────
    print("\n1. GENUINE PAIRS (should output HIGH similarity):")
    print("-" * 70)

    for writer_id in ["1", "2", "3", "4", "5"]:
        genuine_files = sorted(org_path.glob(f"original_{writer_id}_*.png"))

        if len(genuine_files) >= 2:
            img1_path = genuine_files[0]
            img2_path = genuine_files[1]

            img1 = np.expand_dims(preprocess(img1_path), 0)
            img2 = np.expand_dims(preprocess(img2_path), 0)

            prob = float(model.predict([img1, img2], verbose=0)[0][0])

            status = "✓ CORRECT" if prob > 0.5 else "✗ WRONG"
            color  = GREEN if prob > 0.5 else RED

            print(f"Writer {writer_id}: {img1_path.name} + {img2_path.name}")
            print(f"  Probability: {color}{prob:.4f}{RESET} → "
                  f"{'GENUINE' if prob > 0.5 else 'FORGED'} {status}")

    # ─────────────────────────
    # 2️⃣ Forged pairs
    # ─────────────────────────
    print("\n2. FORGED PAIRS (should output LOW similarity):")
    print("-" * 70)

    for writer_id in ["1", "2", "3", "4", "5"]:
        genuine_files = sorted(org_path.glob(f"original_{writer_id}_*.png"))
        forged_files  = sorted(forg_path.glob(f"forgeries_{writer_id}_*.png"))

        if genuine_files and forged_files:
            img1_path = genuine_files[0]
            img2_path = forged_files[0]

            img1 = np.expand_dims(preprocess(img1_path), 0)
            img2 = np.expand_dims(preprocess(img2_path), 0)

            prob = float(model.predict([img1, img2], verbose=0)[0][0])

            status = "✓ CORRECT" if prob < 0.5 else "✗ WRONG"
            color  = GREEN if prob < 0.5 else RED

            print(f"Writer {writer_id}: {img1_path.name} + {img2_path.name}")
            print(f"  Probability: {color}{prob:.4f}{RESET} → "
                  f"{'GENUINE' if prob > 0.5 else 'FORGED'} {status}")

    # ─────────────────────────
    # 3️⃣ Different writers
    # ─────────────────────────
    print("\n3. DIFFERENT WRITERS (should output LOW similarity):")
    print("-" * 70)

    img1_path = next(org_path.glob("original_1_*.png"))
    img2_path = next(org_path.glob("original_2_*.png"))

    img1 = np.expand_dims(preprocess(img1_path), 0)
    img2 = np.expand_dims(preprocess(img2_path), 0)

    prob = float(model.predict([img1, img2], verbose=0)[0][0])

    status = "✓ CORRECT" if prob < 0.5 else "✗ WRONG"
    color  = GREEN if prob < 0.5 else RED

    print(f"Writer 1 vs Writer 2: {img1_path.name} + {img2_path.name}")
    print(f"  Probability: {color}{prob:.4f}{RESET} → "
          f"{'GENUINE' if prob > 0.5 else 'FORGED'} {status}")

    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("  • Genuine pairs should be HIGH (close to 1)")
    print("  • Forged pairs should be LOW (close to 0)")
    print("  • If everything is ~0.5 → model didn’t learn")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
