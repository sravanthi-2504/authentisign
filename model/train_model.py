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
    def __init__(self, dataset_path, batch_size=16, input_shape=(128,128,1), mode="train", **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.input_shape = input_shape

        org_path = Path(dataset_path) / "full_org"
        forg_path = Path(dataset_path) / "full_forg"

        self.writers = {}

        # Collect genuine signatures
        for img_path in org_path.glob("*.png"):
            parts = img_path.stem.split('_')
            if len(parts) < 3:
                continue
            writer_id = parts[1]

            self.writers.setdefault(writer_id, {"genuine": [], "forged": []})
            self.writers[writer_id]["genuine"].append(img_path)


    # Collect forged signatures
        for img_path in forg_path.glob("*.png"):
            parts = img_path.stem.split('_')
            if len(parts) < 3:
                continue
            writer_id = parts[1]

            if writer_id in self.writers:
                self.writers[writer_id]["forged"].append(img_path)
        self.writer_ids = list(self.writers.keys())
        # 🔥 Split 80% train, 20% validation
        split_index = int(len(self.writer_ids) * 0.8)
        self.train_writers = self.writer_ids[:split_index]
        self.val_writers = self.writer_ids[split_index:]

        self.mode = mode

        print(f"Loaded {len(self.writer_ids)} writers")
        print(f"Train writers: {len(self.train_writers)}")
        print(f"Validation writers: {len(self.val_writers)}")
        print(f"Loaded {len(self.writer_ids)} writers from CEDAR dataset")

    def __len__(self):
        return 400  # steps per epoch

    def preprocess(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))

        # Apply thresholding for better contrast
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    def __getitem__(self, index):
        X1, X2, y = [], [], []

        while len(X1) < self.batch_size:
            writer_list = self.train_writers if self.mode == "train" else self.val_writers
            writer = np.random.choice(writer_list)
            genuine = self.writers[writer]["genuine"]
            forged = self.writers[writer]["forged"]

        # Skip invalid writers
            if len(genuine) < 2:
                continue

            if np.random.rand() < 0.5:
            # Genuine pair
                img1, img2 = np.random.choice(genuine, 2, replace=False)
                label = 1
            else:
            # Forged pair
                if len(forged) < 1:
                    continue
                img1 = np.random.choice(genuine)
                img2 = np.random.choice(forged)
                label = 0

            try:
                X1.append(self.preprocess(img1))
                X2.append(self.preprocess(img2))
                y.append(label)
            except:
                continue

        return (np.array(X1), np.array(X2)), np.array(y)



class SignatureVerificationModel:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def create_base_network(self):
        """CNN for feature extraction"""
        input_layer = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        embedding = layers.Dense(128)(x)  # No activation

        return Model(input_layer, embedding)

    def create_siamese_network(self):
        """Create Siamese Network WITHOUT Lambda layers - fully serializable"""
        base_network = self.create_base_network()

        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')

        # Generate embeddings
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)

        # ✅ Custom Layer for absolute value (NO Lambda!)
        class AbsoluteLayer(layers.Layer):
            def call(self, inputs):
                return tf.abs(inputs)

            def get_config(self):
                return super().get_config()

        # Calculate L1 distance
        distance = layers.Subtract()([embedding_a, embedding_b])
        distance = AbsoluteLayer(name='absolute_distance')(distance)

        # Classification head
        x = layers.Dense(64, activation='relu')(distance)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid', name='output')(x)

        self.model = Model([input_a, input_b], output)
        return self.model

    def train(self, data_dir, epochs=30, batch_size=16):
        print("Creating dynamic generator...")

        train_generator = CEDARPairGenerator(
            dataset_path=data_dir,
            batch_size=batch_size,
            input_shape=self.input_shape,
            mode="train"
        )

        val_generator = CEDARPairGenerator(
            dataset_path=data_dir,
            batch_size=batch_size,
            input_shape=self.input_shape,
            mode="val"
        )

        print("\nBuilding model...")
        self.create_siamese_network()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print(self.model.summary())

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]

        print("\nTraining...")
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )


        return self.history

    def save_model(self, path="signature_model_final.h5"):
        """Save model"""
        # Save as .keras (recommended)
        keras_path = path.replace('.h5', '.keras')
        self.model.save(keras_path)
        print(f"✓ Model saved to {keras_path}")

        # Also save weights
        weights_path = path.replace('.h5', '.weights.h5')
        self.model.save_weights(weights_path)
        print(f"✓ Weights saved to {weights_path}")


if __name__ == "__main__":
    print("="*60)
    print("  CEDAR SIGNATURE VERIFICATION - MODEL TRAINING")
    print("="*60)

    dataset_path = "../dataset/cedar_dataset"

    model = SignatureVerificationModel()

    model.train(
        data_dir=dataset_path,
        epochs=45,
        batch_size=16
    )

    model.save_model("signature_model_final.h5")

    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)