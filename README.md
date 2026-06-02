# 🧠 NeuroScan AI — Brain Tumor Detection System

NeuroScan AI is a professional, mobile-responsive Flask web application designed for classifying brain MRI scans into four distinct categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **No Tumor Detected**
- **Pituitary Tumor**

The application leverages a deep learning model based on the **EfficientNetB0** architecture, specifically trained using **Transfer Learning** for clinical image classification, achieving a test accuracy of **94.05%**.

---

## 🚀 Key Features

* **Efficient Local Inference:** Executes model inference locally using a pre-compiled TensorFlow/Keras runtime.
* **Advanced Medical Validation:** Integrated image guardrails check structural features (dark background, central brain structures, grayscale channels) to prevent non-MRI or color images from being processed.
* **Interactive Workbench UI:** A premium, modern interface featuring a dark clinical theme, subtle micro-animations, progress tracking, and custom diagnostic cards.
* **Glassmorphic Lightbox Modal:** Open model training history, confusion matrices, and Grad-CAM attention maps in a responsive full-screen modal view.
* **Fully Responsive:** Layouts adapt seamlessly from widescreen workstations down to ultra-compact mobile screen dimensions.

---

## 🧠 Model Architecture & Training Techniques

NeuroScan AI uses a state-of-the-art Convolutional Neural Network (CNN) engineered for optimal accuracy and training efficiency.

![Model Architecture](static/images/model_architecture.png)


### 1. Model Backbone
- **Architecture:** `EfficientNetB0`
- **Pre-trained Weights:** ImageNet (locally cached)
- **Trainable Status:** Frozen (`backbone.trainable = False`) to preserve rich visual features and expedite training on CPU/GPU.

### 2. Custom Classification Head
To map extracted features to the brain tumor categories, a custom head was appended and trained:
- **Global Pooling:** Global Average Pooling yields a 1280-dimensional feature vector.
- **Regularization:** A `Dropout(0.25)` layer prevents overfitting on training data.
- **Classifier:** A `Dense(4, activation='softmax')` layer produces final class probability spreads.

### 3. Hyperparameters & Optimization
- **Optimizer:** `Adam` optimizer with a learning rate of `1e-3` for fast, stable convergence.
- **Loss Function:** `Sparse Categorical Crossentropy` suited for integer-mapped classification labels.
- **Early Stopping:** Monitored validation accuracy with a patience window of 8 epochs, restoring the best model weights dynamically.
- **Evaluation Performance:** Tested on independent datasets, yielding a **94.05% test accuracy**.

---

## 🛡️ Image Preprocessing & Clinical Guardrails

To prevent incorrect diagnostic signals, the Flask API evaluates every uploaded scan through four automated validation steps before running inference:

| Guardrail Check | Algorithm Metric / Threshold | Rationale |
| :--- | :--- | :--- |
| **Grayscale Check** | Mean color difference between channels < 12.0 | Filters out color photographs and non-medical images. |
| **Dark Border Check** | Average outer 10% border brightness < 45.0 (0-255 range) | Ensures the image contains the dark background typical of MRI scans. |
| **Central Contrast Check** | Average center 50% brightness > 15.0 | Ensures the scan is not blank or pitch black. |
| **Brain Structure Check** | Center brightness > Border brightness + 5.0 | Confirms the presence of a brighter central tissue structure. |

---

## 📁 Project Structure

```text
├── app.py                            # Flask application server, API, and validation logic
├── brain_tumor_model_efficientnet.keras # Saved production TensorFlow model
├── model_metadata.json               # Model metrics (accuracy, epochs, backbone metadata)
├── train_efficientnet_head.py        # Model training and feature extraction pipeline
├── test_model.py                     # Automated model inference validation checks
├── test_validation.py                # Automated image validation and guardrail checks
├── requirements.txt                  # Python application dependencies
├── brain_tumor_dataset/              # Original training & testing dataset split
├── static/
│   ├── index.html                    # Workbench HTML structure & modal layout
│   ├── style.css                     # Main stylesheet with mobile responsive grids
│   ├── script.js                     # File upload, AJAX API client, and DOM interactions
│   └── images/                       # Model evaluation artifacts & visualizations
│       ├── confusion_matrix.png      # Matrix of true vs. predicted model outputs
│       ├── training_history.png      # Training/Validation accuracy and loss curves
│       └── gradcam_brain_tumor.png   # Model activation overlays indicating focus areas
└── uploads/                          # Temporary workspace directory for file uploads
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- TensorFlow 2.10+ (included in `requirements.txt`)

### Step 1: Clone and Install Dependencies
Install all required libraries within your Python environment or virtual environment:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Web Server
Launch the application locally:
```bash
python app.py
```
The console will confirm that the model loaded successfully and show the server URL:
```text
============================================================
Brain Tumor Detection - Web Application
============================================================

Starting server...
Model loaded: Yes
Open your browser and go to: http://localhost:5000
============================================================
```

---

## 🧪 Testing

The codebase includes automated test suites to ensure model loading, inference correctness, and validation rules remain reliable.

### Run Model Inference Tests
```bash
python test_model.py
```
*Validates that the model correctly classifies baseline tumor patterns.*

### Run Guardrail Validation Tests
```bash
python test_validation.py
```
*Validates that the image check rules block color images, empty black images, and non-MRI files.*

---

## ⚠️ Clinical Disclaimer

> [!WARNING]
> This system is designed solely as an **educational prototype and research signal**. The predictions generated by the EfficientNetB0 model should not be used as professional medical advice, clinical diagnosis, treatment planning, or to make patient-care decisions. All diagnosis must be performed by qualified healthcare professionals using official, certified medical diagnostics.
