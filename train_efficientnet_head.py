"""
Train a lightweight classifier head on cached EfficientNetB0 features.

This is much faster than full CNN training on CPU because the ImageNet
backbone is frozen and each dataset image is passed through it once.
"""

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "brain_tumor_dataset" / "Training"
TEST_DIR = BASE_DIR / "brain_tumor_dataset" / "Testing"
OUTPUT_MODEL = BASE_DIR / "brain_tumor_model_efficientnet.keras"
OUTPUT_METADATA = BASE_DIR / "model_metadata.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def build_dataset(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def extract_features(backbone, dataset, label):
    features = []
    labels = []
    total_batches = tf.data.experimental.cardinality(dataset).numpy()

    print(f"Extracting {label} features from {total_batches} batches...")
    for index, (images, batch_labels) in enumerate(dataset, start=1):
        batch_features = backbone(images, training=False).numpy()
        features.append(batch_features)
        labels.append(batch_labels.numpy())

        if index == 1 or index == total_batches or index % 20 == 0:
            print(f"  {label}: {index}/{total_batches} batches")

    return np.concatenate(features), np.concatenate(labels)


def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError("Expected brain_tumor_dataset/Training and Testing directories.")

    print("Loading datasets...")
    train_ds = build_dataset(TRAIN_DIR)
    test_ds = build_dataset(TEST_DIR)

    print("Loading cached EfficientNetB0 ImageNet backbone...")
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False

    x_train, y_train = extract_features(backbone, train_ds, "train")
    x_test, y_test = extract_features(backbone, test_ds, "test")

    print("Training classifier head...")
    head_input = keras.Input(shape=(x_train.shape[1],), name="efficientnet_features")
    x = layers.Dropout(0.25)(head_input)
    output = layers.Dense(len(CLASS_NAMES), activation="softmax", name="tumor_class")(x)
    head = keras.Model(head_input, output, name="BrainTumorEfficientNetHead")
    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = head.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = head.evaluate(x_test, y_test, verbose=0)
    print(f"EfficientNet head test accuracy: {test_accuracy:.4f}")

    full_input = keras.Input(shape=(*IMG_SIZE, 3), name="mri_image")
    # The Flask app normalizes images to 0..1. EfficientNetB0 expects 0..255.
    scaled = layers.Rescaling(255.0, name="restore_pixel_range")(full_input)
    features = backbone(scaled, training=False)
    full_output = head(features, training=False)
    full_model = keras.Model(full_input, full_output, name="BrainTumorEfficientNetB0")
    full_model.save(OUTPUT_MODEL)

    metadata = {
        "model_file": OUTPUT_MODEL.name,
        "backbone": "EfficientNetB0 ImageNet",
        "class_names": CLASS_NAMES,
        "image_size": IMG_SIZE,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "epochs_run": len(history.history.get("loss", [])),
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {OUTPUT_MODEL}")
    print(f"Saved metadata: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
