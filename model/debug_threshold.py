import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# debug_threshold.py
import numpy as np
import tensorflow as tf
import cv2
import os
from model.train_model import L2Normalization

from tensorflow.keras.layers import Layer

class AbsoluteLayer(Layer):
    def __init__(self, **kwargs):
        super(AbsoluteLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.abs(inputs)

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend", "signature_model_final.keras")
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={
        "L2Normalization": L2Normalization,
        "AbsoluteLayer": AbsoluteLayer
    }
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "cedar_dataset")
print("Model inputs:", len(model.inputs))
print("Model output shape:", model.output_shape)

def preprocess(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128, 128))
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, -1)

def get_distance(p1, p2):
    i1 = np.expand_dims(preprocess(p1), 0)
    i2 = np.expand_dims(preprocess(p2), 0)

    # Siamese model takes BOTH inputs together
    prediction = model.predict([i1, i2])

    # If model outputs similarity score (sigmoid)
    return float(prediction)

# ✅ Put paths to your known genuine pairs here
genuine_pairs = [
    (
        os.path.join(DATASET_DIR, "full_org", "original_18_1.png"),
        os.path.join(DATASET_DIR, "full_org", "original_18_2.png"),
    ),
    (
        os.path.join(DATASET_DIR, "full_org", "original_29_1.png"),
        os.path.join(DATASET_DIR, "full_org", "original_29_2.png"),
    ),
]

forged_pairs = [
    (
        os.path.join(DATASET_DIR, "full_org", "original_18_1.png"),
        os.path.join(DATASET_DIR, "full_forg", "forgeries_18_1.png"),
    ),
    (
        os.path.join(DATASET_DIR, "full_org", "original_29_1.png"),
        os.path.join(DATASET_DIR, "full_forg", "forgeries_29_1.png"),
    ),
]

print("\n--- GENUINE PAIRS (distances should be LOW) ---")
for p1, p2 in genuine_pairs:
    d = get_distance(p1, p2)
    print(f"  Distance: {d:.4f}")

print("\n--- FORGED PAIRS (distances should be HIGH) ---")
for p1, p2 in forged_pairs:
    d = get_distance(p1, p2)
    print(f"  Distance: {d:.4f}")

print("\n→ Set your threshold between the two groups above")