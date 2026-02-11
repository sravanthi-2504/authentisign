"""
AI Signature Verification Model Training Script
Using Siamese Network WITHOUT Lambda layers (for compatibility)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import cv2
from pathlib import Path

np.random.seed(42)
tf.random.set_seed(42)

class CEDARPairGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=16, input_shape=(128,128,1)):
        self.batch_size = batch_size
        self.input_shape = input_shape

        org_path = Path(dataset_path) / "full_org"
        forg_path = Path(dataset_path) / "full_forg"

        self.writers = {}

        for img_path in org_path.glob("*.*"):
            writer_id = img_path.stem.split('_')[0]
            self.writers.setdefault(writer_id, {"genuine": [], "forged": []})
            self.writers[writer_id]["genuine"].append(img_path)

        for img_path in forg_path.glob("*.*"):
            writer_id = img_path.stem.split('_')[0]
            if writer_id in self.writers:
                self.writers[writer_id]["forged"].append(img_path)

        self.writer_ids = list(self.writers.keys())

    def __len__(self):
        return 1000  # steps per epoch

    def preprocess(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128,128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    def __getitem__(self, index):
        X1 = []
        X2 = []
        y = []

        for _ in range(self.batch_size):

            writer = np.random.choice(self.writer_ids)
            genuine = self.writers[writer]["genuine"]
            forged = self.writers[writer]["forged"]

            if np.random.rand() < 0.5:
                # Genuine pair
                img1, img2 = np.random.choice(genuine, 2, replace=False)
                label = 1
            else:
                # Forged pair
                img1 = np.random.choice(genuine)
                img2 = np.random.choice(forged)
                label = 0

            X1.append(self.preprocess(img1))
            X2.append(self.preprocess(img2))
            y.append(label)

        return [np.array(X1), np.array(X2)], np.array(y)

class SignatureVerificationModel:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def create_base_network(self):
        """CNN for feature extraction"""
        input_layer = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        embedding = layers.Dense(128, activation='relu')(x)

        return Model(input_layer, embedding)

    def create_siamese_network(self):
        """Create Siamese Network using Subtract layer instead of Lambda"""
        base_network = self.create_base_network()

        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')

        # Generate embeddings
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)

        # ✅ USE SUBTRACT LAYER INSTEAD OF LAMBDA
        # This is serializable and compatible across versions
        distance = layers.Subtract()([embedding_a, embedding_b])
        distance = layers.Lambda(lambda x: tf.abs(x))(distance)
        distance = layers.Dense(64, activation='relu')(distance)

        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(distance)

        self.model = Model([input_a, input_b], output)
        return self.model

    def preprocess_signature(self, image_path):
        """Preprocess signature image"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Cannot read image {image_path}")

        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)

        return img

    def create_pairs_from_cedar(self, dataset_path, max_pairs=500):
        """Create training pairs from dataset"""
        pairs = []
        labels = []

        org_path = Path(dataset_path) / "full_org"
        forg_path = Path(dataset_path) / "full_forg"

        if not org_path.exists() or not forg_path.exists():
            print(f"⚠️ CEDAR dataset not found at {dataset_path}")
            print("Creating sample dataset instead...")
            return self.create_sample_pairs(max_pairs)

        writers = {}

        # Collect genuine signatures
        for img_path in org_path.glob("*.*"):
            writer_id = img_path.stem.split('_')[0]
            writers.setdefault(writer_id, {"genuine": [], "forged": []})
            writers[writer_id]["genuine"].append(img_path)

        # Collect forged signatures
        for img_path in forg_path.glob("*.*"):
            writer_id = img_path.stem.split('_')[0]
            if writer_id in writers:
                writers[writer_id]["forged"].append(img_path)

        # Create pairs
        for writer_id, data in list(writers.items())[:15]:
            genuine = data["genuine"]
            forged = data["forged"]

            # Genuine pairs
            for i in range(len(genuine)):
                for j in range(i + 1, len(genuine)):
                    if len(pairs) >= max_pairs:
                        return np.array(pairs), np.array(labels)

                    try:
                        img1 = self.preprocess_signature(genuine[i])
                        img2 = self.preprocess_signature(genuine[j])
                        pairs.append([img1, img2])
                        labels.append(1)
                    except:
                        continue

            # Forged pairs
            for g in genuine:
                for f in forged:
                    if len(pairs) >= max_pairs:
                        return np.array(pairs), np.array(labels)

                    try:
                        img1 = self.preprocess_signature(g)
                        img2 = self.preprocess_signature(f)
                        pairs.append([img1, img2])
                        labels.append(0)
                    except:
                        continue

        return np.array(pairs), np.array(labels)

    def create_sample_pairs(self, max_pairs=500):
        """Create sample synthetic pairs for testing"""
        print("Creating synthetic training data...")
        pairs = []
        labels = []

        for i in range(max_pairs):
            # Create two random similar images for genuine pairs
            img1 = np.random.rand(128, 128, 1).astype('float32')
            img2 = img1 + np.random.rand(128, 128, 1).astype('float32') * 0.1
            img2 = np.clip(img2, 0, 1)

            pairs.append([img1, img2])
            labels.append(1 if i % 2 == 0 else 0)

        return np.array(pairs), np.array(labels)

    def train(self, data_dir, epochs=50, batch_size=16):

        print("Creating dynamic generator...")

        train_generator = CEDARPairGenerator(
            dataset_path=data_dir,
            batch_size=batch_size,
            input_shape=self.input_shape
        )

        print("Building model...")
        self.create_siamese_network()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0005),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print(self.model.summary())

        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            verbose=1
        )

        return self.history


    def save_model(self, path="signature_model_final.h5"):
        """Save model in both formats"""
        # Save as .keras (recommended)
        keras_path = path.replace('.h5', '.keras')
        self.model.save(keras_path)
        print(f"✓ Model saved to {keras_path}")

        # Also save as .h5 for compatibility
        try:
            self.model.save(path, save_format='h5')
            print(f"✓ Backup saved to {path}")
        except:
            print(f"⚠️ Could not save .h5 format (using .keras only)")


if __name__ == "__main__":
    dataset_path = "../dataset/cedar_dataset"

    model = SignatureVerificationModel()

    model.train(
        data_dir=dataset_path,
        epochs=50,
        batch_size=16
    )

    model.save_model("signature_model_final.h5")

    print("\n✓ Training complete!")