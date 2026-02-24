
import numpy as np

import tensorflow as tf

import cv2

import sys

import os

sys.path.append(os.path.abspath("."))

from model.train_model import L2Normalization

model = tf.keras.models.load_model(

    "model/signature_embedding_model.keras",

    compile=False, safe_mode=False,

    custom_objects={"L2Normalization": L2Normalization}

)

# Warm up

_ = model(tf.zeros((1,128,128,1)), training=False)

def preprocess(path):

    img = cv2.imread(path, 0)

    img = cv2.resize(img, (128,128))

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    img = img.astype("float32")/255.0

    return np.expand_dims(img,-1)

def dist(p1, p2):

    i1 = np.expand_dims(preprocess(p1),0)

    i2 = np.expand_dims(preprocess(p2),0)

    e1 = model(i1, training=False).numpy()[0]

    e2 = model(i2, training=False).numpy()[0]

    return float(np.sqrt(np.sum((e1-e2)**2)))

print("\n=== GENUINE vs GENUINE (should be LOW) ===")

for i in range(1,6):

    for j in range(i+1,6):

        p1 = f"dataset/cedar_dataset/full_org/original_29_{i}.png"

        p2 = f"dataset/cedar_dataset/full_org/original_29_{j}.png"

        if os.path.exists(p1) and os.path.exists(p2):

            print(f"  org {i} vs org {j}: {dist(p1,p2):.4f}")

print("\n=== GENUINE vs FORGED (should be HIGH) ===")

for i in range(1,6):

    p1 = f"dataset/cedar_dataset/full_org/original_29_1.png"

    p2 = f"dataset/cedar_dataset/full_forg/forgeries_29_{i}.png"

    if os.path.exists(p1) and os.path.exists(p2):

        print(f"  org vs forg {i}: {dist(p1,p2):.4f}")

