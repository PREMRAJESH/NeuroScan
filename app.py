"""
Brain Tumor Detection - Flask Web Application.

Provides a web interface for uploading MRI scans and getting predictions using
ONNX Runtime for fast, lightweight local inference.
"""

import os
from html import escape
from pathlib import Path
from uuid import uuid4
import json
import io

import numpy as np
import onnxruntime as rt
from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
STATIC_FOLDER = BASE_DIR / "static"

# Use /tmp for uploads on Vercel (read-only filesystem), local folder otherwise
IS_VERCEL = os.environ.get("VERCEL") == "1"
UPLOAD_FOLDER = Path("/tmp/uploads") if IS_VERCEL else BASE_DIR / "uploads"
DATASET_FOLDER = BASE_DIR / "brain_tumor_dataset"
MODEL_METADATA_PATH = BASE_DIR / "model_metadata.json"

# ONNX model configuration
ONNX_MODEL_PATH = BASE_DIR / "brain_tumor_model_efficientnet.onnx"

# Create upload folder only if not on Vercel (which has read-only filesystem)
if not IS_VERCEL:
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    STATIC_FOLDER.mkdir(exist_ok=True)
else:
    # On Vercel/Lambda, /tmp is writable but may not exist yet
    UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DETAILS = {
    "glioma": {
        "label": "Glioma Tumor",
        "description": "A tumor pattern commonly associated with glial tissue findings in MRI scans.",
        "sample": "Testing/glioma/Te-gl_0010.jpg",
    },
    "meningioma": {
        "label": "Meningioma Tumor",
        "description": "A tumor pattern often seen near the membranes surrounding the brain and spinal cord.",
        "sample": "Testing/meningioma/Te-me_0010.jpg",
    },
    "notumor": {
        "label": "No Tumor Detected",
        "description": "A scan pattern classified by the model as not showing one of the trained tumor classes.",
        "sample": "Testing/notumor/Te-no_0010.jpg",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "description": "A tumor pattern commonly located around the pituitary region in MRI images.",
        "sample": "Testing/pituitary/Te-pi_0010.jpg",
    },
}

ARTIFACT_FILES = {
    "confusion-matrix": "static/images/confusion_matrix.png",
    "training-history": "static/images/training_history.png",
    "gradcam": "static/images/gradcam_brain_tumor.png",
}


def build_sample_placeholder(class_name, label):
        """Return a lightweight inline placeholder when dataset samples are absent."""
        safe_class_name = escape(class_name)
        safe_label = escape(label)
        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='720' height='480' viewBox='0 0 720 480' role='img' aria-label='{safe_label} sample unavailable'>
    <defs>
        <linearGradient id='bg' x1='0%' y1='0%' x2='100%' y2='100%'>
            <stop offset='0%' stop-color='#0f1722'/>
            <stop offset='100%' stop-color='#1f3042'/>
        </linearGradient>
    </defs>
    <rect width='720' height='480' rx='28' fill='url(#bg)'/>
    <rect x='56' y='56' width='608' height='368' rx='24' fill='none' stroke='#4b667d' stroke-width='2' stroke-dasharray='10 10' opacity='0.8'/>
    <text x='50%' y='44%' fill='#f2f6fb' font-family='Arial, Helvetica, sans-serif' font-size='34' font-weight='700' text-anchor='middle'>{safe_label}</text>
    <text x='50%' y='53%' fill='#a9bbca' font-family='Arial, Helvetica, sans-serif' font-size='20' text-anchor='middle'>Sample image not bundled in deployment</text>
    <text x='50%' y='61%' fill='#8aa0b3' font-family='Arial, Helvetica, sans-serif' font-size='16' text-anchor='middle'>Class key: {safe_class_name}</text>
</svg>"""
        return Response(svg, mimetype="image/svg+xml")


def allowed_file(filename):
    """Check whether a filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image_for_onnx(img_path):
    """Preprocess image for ONNX Runtime inference.
    
    Converts PIL image to normalized numpy array matching EfficientNetB0 requirements:
    - Resized to 224x224
    - Normalized to [-1, 1] range (ImageNet preprocessing)
    - NHWC format (batch, height, width, channels)
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1] range (EfficientNet standard)
        img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension: HWC -> BHWC (no transpose needed - keep channels-last)
        img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def validate_mri_image(img_path):
    """
    Validate whether the uploaded image is likely a brain MRI scan.
    Returns (True, None) if valid, or (False, "error message") if it's a non-MRI scan.
    """
    from PIL import Image

    try:
        with Image.open(img_path) as img:
            # Convert to RGB to analyze color channels
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)
            
            # 1. Check for color saturation (brain MRIs are grayscale images)
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
            mean_color_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) / 3.0
            
            # Standard color photos have a mean color difference > 12.0.
            # Real MRIs will have a difference close to 0 (subtle compression artifacts might be 1.0-3.0).
            if mean_color_diff > 12.0:
                return False, "The uploaded file appears to be a color image. Brain MRI scans must be grayscale images. Please upload a valid brain MRI scan."
            
            # 2. Check for a dark background (real MRIs are centered against dark empty space)
            h, w, _ = img_np.shape
            border_h = max(1, int(h * 0.10))
            border_w = max(1, int(w * 0.10))
            
            # Extract border regions (outer 10% margins)
            top_border = img_np[0:border_h, :, :]
            bottom_border = img_np[h-border_h:h, :, :]
            left_border = img_np[border_h:h-border_h, 0:border_w, :]
            right_border = img_np[border_h:h-border_h, w-border_w:w, :]
            
            border_pixels = np.concatenate([
                top_border.flatten(),
                bottom_border.flatten(),
                left_border.flatten(),
                right_border.flatten()
            ])
            
            mean_border_val = np.mean(border_pixels)
            
            # In a medical MRI scan, the border is almost completely black.
            # Normal photos, documents, or uniform bright backgrounds will have bright borders.
            if mean_border_val > 45.0:
                return False, "The uploaded image does not have the dark background typical of a brain MRI scan. Please upload a valid brain MRI scan."
            
            # 3. Check that there is actually a central structure (not just a solid black/dark image)
            center_h_start = int(h * 0.25)
            center_h_end = int(h * 0.75)
            center_w_start = int(w * 0.25)
            center_w_end = int(w * 0.75)
            center_pixels = img_np[center_h_start:center_h_end, center_w_start:center_w_end, :]
            
            mean_center_val = np.mean(center_pixels)
            
            # If the center is also completely pitch black, it's not a valid scan.
            if mean_center_val < 15.0:
                return False, "The uploaded image is too dark or empty. Please upload a valid brain MRI scan."
            
            # 4. Check if the center has a brain structure (should be brighter than the empty border space)
            if mean_center_val <= mean_border_val + 5.0:
                return False, "The uploaded image does not show a centered brain structure with a dark background. Please upload a valid brain MRI scan."
            
            return True, None
            
    except Exception as exc:
        return False, f"Failed to validate image format: {str(exc)}"


def get_friendly_name(class_name):
    """Convert an internal class key to a user-facing label."""
    return CLASS_DETAILS.get(class_name, {}).get("label", class_name)


def load_prediction_model():
    """Load ONNX Runtime inference session for the model."""
    if not ONNX_MODEL_PATH.exists():
        return None, f"Model file not found: {ONNX_MODEL_PATH}"
    
    try:
        # Create ONNX Runtime session with CPU execution provider
        session = rt.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        return session, None
    except Exception as exc:
        return None, f"Failed to load ONNX model: {str(exc)}"


def load_model_metadata():
    """Read optional metadata written by the transfer-learning repair script."""
    if not MODEL_METADATA_PATH.exists():
        return {}

    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        return metadata
    except Exception as exc:
        print(f"Model metadata could not be read: {exc}")
        return {}


print("Loading model...")
model_metadata = load_model_metadata()
model, model_load_error = load_prediction_model()
if model_load_error:
    print(f"Model failed to load: {model_load_error}")
else:
    print("Model loaded successfully.")


@app.route("/")
def index():
    """Serve the main page."""
    return send_from_directory(STATIC_FOLDER, "index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static frontend files."""
    return send_from_directory(STATIC_FOLDER, filename)


@app.route("/artifacts/<artifact_key>")
def serve_artifact(artifact_key):
    """Serve model evaluation images used by the frontend."""
    filename = ARTIFACT_FILES.get(artifact_key)
    if not filename:
        return jsonify({"error": "Artifact not found"}), 404
    return send_from_directory(BASE_DIR, filename)


@app.route("/samples/<class_name>")
def serve_sample(class_name):
    """Serve one representative dataset image for a supported class."""
    detail = CLASS_DETAILS.get(class_name)
    if not detail:
        return jsonify({"error": "Sample class not found"}), 404

    sample_path = DATASET_FOLDER / detail["sample"]
    if sample_path.exists():
        return send_from_directory(DATASET_FOLDER, detail["sample"])

    return build_sample_placeholder(class_name, detail["label"])


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model and dataset metadata for the app pages."""
    artifacts = {
        key: {
            "url": f"/artifacts/{key}",
            "available": (BASE_DIR / filename).exists(),
        }
        for key, filename in ARTIFACT_FILES.items()
    }

    classes = [
        {
            "key": class_name,
            "label": CLASS_DETAILS[class_name]["label"],
            "description": CLASS_DETAILS[class_name]["description"],
            "sample_url": f"/samples/{class_name}",
        }
        for class_name in CLASS_NAMES
    ]

    return jsonify(
        {
            "model_loaded": model is not None,
            "model_error": model_load_error,
            "model_file": "brain-tumor-model-efficientnet.onnx",
            "model_metadata": model_metadata,
            "image_size": IMG_SIZE,
            "classes": classes,
            "artifacts": artifacts,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """Handle prediction requests using ONNX Runtime local inference."""
    if model is None:
        return jsonify({"error": model_load_error or "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Invalid file type. Please upload JPG, JPEG, or PNG."}), 400

    filepath = None
    try:
        safe_name = secure_filename(uploaded_file.filename)
        filepath = UPLOAD_FOLDER / f"{uuid4().hex}_{safe_name}"
        uploaded_file.save(filepath)

        # Validate that the uploaded image is indeed a brain MRI scan
        is_mri, err_msg = validate_mri_image(filepath)
        if not is_mri:
            return jsonify({"error": err_msg}), 400

        # Preprocess image for ONNX Runtime
        img_array = prepare_image_for_onnx(filepath)

        # Run inference with ONNX Runtime
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        
        output_data = model.run([output_name], {input_name: img_array})[0]
        
        # Parse ONNX output: apply softmax to get probabilities
        output_probs = np.exp(output_data - np.max(output_data)) / np.sum(np.exp(output_data - np.max(output_data)), axis=1)
        
        # Get predictions for all classes
        predictions = output_probs[0]  # Take first (only) sample from batch
        
        # Create class-score mapping
        class_scores = {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Find top prediction
        predicted_class = max(class_scores, key=class_scores.get)
        confidence = class_scores[predicted_class]
        
        # Sort all probabilities
        all_probabilities = {
            class_name: round(score, 6)
            for class_name, score in sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        }

        return jsonify(
            {
                "success": True,
                "prediction": get_friendly_name(predicted_class),
                "class_key": predicted_class,
                "confidence": round(confidence, 6),
                "all_probabilities": all_probabilities,
            }
        )

    except Exception as exc:
        print(f"Error during prediction: {exc}")
        return jsonify({"error": str(exc)}), 500
    finally:
        if filepath and filepath.exists():
            filepath.unlink(missing_ok=True)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy" if model is not None else "degraded",
            "model_loaded": model is not None,
            "model_error": model_load_error,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"

    print("\n" + "=" * 60)
    print("Brain Tumor Detection - Web Application")
    print("=" * 60)
    print("\nStarting server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Model loaded: {'Yes' if model else 'No'}")
    if model_load_error:
        print(f"Model error: {model_load_error}")
    print(f"Debug mode: {'On' if debug_mode else 'Off'}")
    print(f"\nOpen your browser and go to: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(debug=debug_mode, host="0.0.0.0", port=port, use_reloader=debug_mode)
