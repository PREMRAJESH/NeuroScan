#!/usr/bin/env python3
"""Convert Keras model to ONNX format for use with ONNX Runtime."""

import tensorflow as tf
import numpy as np
import tempfile
import os
import subprocess
from pathlib import Path

def convert_keras_to_onnx():
    """Convert brain_tumor_model_efficientnet.keras to ONNX format."""
    
    model_path = Path("brain_tumor_model_efficientnet.keras")
    onnx_path = Path("brain_tumor_model_efficientnet.onnx")
    
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded. Input shape: {model.input_shape}")
    
    # Save as SavedModel format (required for tf2onnx)
    temp_dir = tempfile.mkdtemp()
    saved_model_path = os.path.join(temp_dir, 'saved_model')
    print(f"Saving to SavedModel format: {saved_model_path}")
    model.export(saved_model_path)
    
    # Convert to ONNX
    print("Converting to ONNX format...")
    result = subprocess.run([
        'python', '-m', 'tf2onnx.convert',
        '--saved-model', saved_model_path,
        '--output', str(onnx_path),
        '--opset', '13'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Conversion failed: {result.stderr}")
        return False
    
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"✅ ONNX conversion successful!")
        print(f"   Output file: {onnx_path}")
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ ONNX file not created")
        return False

if __name__ == "__main__":
    convert_keras_to_onnx()
