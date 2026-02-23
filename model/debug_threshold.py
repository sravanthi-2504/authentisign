import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# debug_threshold.py
import numpy as np
import tensorflow as tf
import cv2
import os
from model.train_model import L2Normalization

MODEL_PATH = "model/signature_embedding_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH, compile=False, safe_mode=False,
    custom_objects={"L2Normalization": L2Normalization}
)

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
    e1 = model.predict(i1, verbose=0)[0]
    e2 = model.predict(i2, verbose=0)[0]
    return float(np.sqrt(np.sum((e1 - e2) ** 2)))

# ✅ Put paths to your known genuine pairs here
genuine_pairs = [
    ("dataset/cedar_dataset/full_org/original_18_1.png",
     "dataset/cedar_dataset/full_org/original_18_2.png"),

    ("dataset/cedar_dataset/full_org/original_29_1.png",
     "dataset/cedar_dataset/full_org/original_29_2.png"),
]

forged_pairs = [
    ("dataset/cedar_dataset/full_org/original_18_1.png",
     "dataset/cedar_dataset/full_forg/forgeries_18_1.png"),

    ("dataset/cedar_dataset/full_org/original_29_1.png",
     "dataset/cedar_dataset/full_forg/forgeries_29_1.png"),
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