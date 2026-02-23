"""
Find Best Threshold for Signature Verification
Run AFTER training is completed
"""

import numpy as np
import tensorflow as tf
from train_model import PairGenerator
from train_model import L2Normalization

DATASET_PATH = "../dataset/cedar_dataset"
MODEL_PATH = "signature_embedding_model.keras"

print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={"L2Normalization": L2Normalization}
)

print("Loading validation generator...")
val_gen = PairGenerator(DATASET_PATH, batch_size=32, mode="validation")

distances = []
labels = []

print("Collecting distances...")

for i in range(len(val_gen)):
    (img1, img2), y = val_gen[i]

    # Get embeddings separately
    e1 = model.predict(img1, verbose=0)
    e2 = model.predict(img2, verbose=0)

    # Compute L2 distance manually
    d = np.sqrt(np.sum((e1 - e2) ** 2, axis=1))

    distances.extend(d)
    labels.extend(y)

distances = np.array(distances)
labels = np.array(labels)

print("Searching best threshold...")

best_acc = 0
best_thresh = 0

for t in np.linspace(0.2, 1.2, 400):
    preds = distances < t
    acc = np.mean(preds == labels)

    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("\n" + "="*60)
print("BEST THRESHOLD:", round(best_thresh, 4))
print("VALIDATION ACCURACY:", round(best_acc * 100, 2), "%")
print("="*60)