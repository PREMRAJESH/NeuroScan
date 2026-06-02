"""
Test script for verifying MRI scan validation logic.
Tests real brain MRI scans from the dataset as well as programmatically generated
non-MRI images to ensure correctness (zero false positives, zero false negatives).
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add root folder to path so we can import validate_mri_image
sys.path.append(str(Path(__file__).parent))
from app import validate_mri_image

def generate_non_mri_images():
    """Generate temporary non-MRI images for verification."""
    temp_dir = Path(__file__).parent / "temp_test_images"
    temp_dir.mkdir(exist_ok=True)
    
    # 1. A colorful image (non-MRI)
    color_img = Image.new("RGB", (224, 224))
    pixels = color_img.load()
    for x in range(224):
        for y in range(224):
            # Rich colors: Red gradient and Blue gradient
            pixels[x, y] = (x, y, 100)
    color_path = temp_dir / "test_color_photo.jpg"
    color_img.save(color_path)
    
    # 2. A bright uniform / document-like image
    document_img = Image.new("RGB", (224, 224), color=(240, 240, 245))
    doc_path = temp_dir / "test_document.jpg"
    document_img.save(doc_path)
    
    # 3. A solid pitch black image
    black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    black_path = temp_dir / "test_solid_black.jpg"
    black_img.save(black_path)
    
    # 4. A solid grey image
    grey_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    grey_path = temp_dir / "test_solid_grey.jpg"
    grey_img.save(grey_path)
    
    return {
        "color": color_path,
        "document": doc_path,
        "black": black_path,
        "grey": grey_path
    }

def clean_temp_images():
    """Remove generated test files."""
    temp_dir = Path(__file__).parent / "temp_test_images"
    if temp_dir.exists():
        for file in temp_dir.iterdir():
            file.unlink()
        temp_dir.rmdir()

def main():
    print("=" * 60)
    print("Verification of validate_mri_image Logic")
    print("=" * 60)
    
    # Step 1: Generate non-MRI test images
    non_mri_files = generate_non_mri_images()
    
    success = True
    
    # Step 2: Test non-MRI rejections (expecting False)
    print("\n--- Testing Non-MRI Image Rejections (Expecting Rejection) ---")
    for key, path in non_mri_files.items():
        is_mri, err_msg = validate_mri_image(path)
        if not is_mri:
            print(f"[PASS] Successfully rejected {key} image. Error: '{err_msg}'")
        else:
            print(f"[FAIL] Erroneously accepted {key} image as a valid MRI!")
            success = False
            
    # Step 3: Test real MRI scan acceptances (expecting True)
    print("\n--- Testing Real MRI Scan Acceptances (Expecting Acceptance) ---")
    dataset_testing_dir = Path(__file__).parent / "brain_tumor_dataset" / "Testing"
    real_mri_samples = []
    
    if dataset_testing_dir.exists():
        # Collect a sample image from each class folder
        for class_dir in dataset_testing_dir.iterdir():
            if class_dir.is_dir():
                first_img = next(class_dir.glob("*.jpg"), None)
                if first_img:
                    real_mri_samples.append((class_dir.name, first_img))
    
    if not real_mri_samples:
        print("[WARNING] No real MRI scans found in dataset. Skipping real scan acceptance tests.")
    else:
        for class_name, path in real_mri_samples:
            is_mri, err_msg = validate_mri_image(path)
            if is_mri:
                print(f"[PASS] Successfully accepted real {class_name} brain MRI: {path.name}")
            else:
                print(f"[FAIL] Erroneously rejected real {class_name} brain MRI! Error: {err_msg}")
                success = False
                
    # Clean up generated images
    clean_temp_images()
    
    print("\n" + "=" * 60)
    if success:
        print("VERIFICATION COMPLETED SUCCESSFULLY: ALL TESTS PASSED!")
    else:
        print("VERIFICATION COMPLETED WITH FAILURES. PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    main()
