"""
Debug script to test model predictions against known dataset samples.
"""

import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = Path(__file__).parent
MODEL_CANDIDATES = [
    BASE_DIR / "brain_tumor_model_efficientnet.keras",
    BASE_DIR / "brain_tumor_model_final.h5",
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[-1])
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

TEST_IMAGES = {
    "glioma": BASE_DIR / "brain_tumor_dataset" / "Testing" / "glioma" / "Te-gl_0010.jpg",
    "meningioma": BASE_DIR / "brain_tumor_dataset" / "Testing" / "meningioma" / "Te-me_0010.jpg",
    "pituitary": BASE_DIR / "brain_tumor_dataset" / "Testing" / "pituitary" / "Te-pi_0010.jpg",
    "notumor": BASE_DIR / "brain_tumor_dataset" / "Testing" / "notumor" / "Te-no_0010.jpg",
}


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Model loaded: {MODEL_PATH.name}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    print("\n" + "=" * 60)
    print("Testing model predictions")
    print("=" * 60)

    correct = 0
    tested = 0

    for class_label, img_path in TEST_IMAGES.items():
        if not img_path.exists():
            print(f"\n[missing] {class_label}: image not found at {img_path}")
            continue

        predictions = model.predict(prepare_image(img_path), verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        tested += 1
        is_correct = predicted_class == class_label
        correct += int(is_correct)

        print(f"\nSample: {class_label.upper()}")
        print(f"   Predicted: {predicted_class} ({confidence:.2%})")
        print(f"   Raw predictions: {predictions[0]}")
        print("   All probabilities:")

        for i, class_name in enumerate(CLASS_NAMES):
            print(f"      [{i}] {class_name}: {predictions[0][i]:.4f}")

        result = "CORRECT" if is_correct else "WRONG"
        print(f"   Result: {result}")

    print("\n" + "=" * 60)
    print(f"Smoke-test accuracy: {correct}/{tested}")


if __name__ == "__main__":
    main()
