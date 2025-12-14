# -*- encoding: UTF-8 -*-
# Vision Offline Test with YOLOv3-Tiny (COCO 80 classes) + Number OCR via Tesseract
# Analyzes existing images from ./captured_images
# No NAO connection needed – detection runs on laptop only

import sys
import time
import cv2
import numpy as np
import os
from tesserocr import PyTessBaseAPI  # For OCR (Python 2.7 compatible)
from PIL import Image  # For image conversion to OCR

# Model setup – place files in ./models/
MODEL_DIR = "./models"
CFG_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
NAMES_FILE = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to report
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold
OCR_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for OCR numbers (30%)


# Load COCO class names (80 classes)
def load_classes():
    if not os.path.exists(NAMES_FILE):
        print("Error: {} not found.".format(NAMES_FILE))
        print(
            "Download from: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        )
        print("Place in ./models/")
        return None
    with open(NAMES_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("Loaded {} COCO classes.".format(len(classes)))
    return classes


CLASSES = load_classes()
if CLASSES is None:
    sys.exit(1)


def load_detection_model():
    """Load YOLOv3-Tiny model (Darknet format)"""
    if not os.path.exists(CFG_FILE):
        print("Error: {} not found.".format(CFG_FILE))
        print(
            "Download from: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        )
        return None
    if not os.path.exists(WEIGHTS_FILE):
        print("Error: {} not found.".format(WEIGHTS_FILE))
        print("Download from: https://pjreddie.com/media/files/yolov3-tiny.weights")
        return None

    net = cv2.dnn.readNetFromDarknet(CFG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU (works on all laptops)
    print("YOLOv3-Tiny model loaded successfully (80 COCO classes).")
    return net


def detect_objects(image, net, target_objects=None):
    """Run YOLOv3-Tiny detection and return filtered list"""
    (h, w) = image.shape[:2]

    # Create blob (YOLO expects 416x416 input)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Forward pass
    outputs = net.forward(output_layers)

    detected = []
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                label = CLASSES[class_id]

                # Filter by target objects if specified
                if target_objects is not None and label not in target_objects:
                    continue

                # Bounding box (center x, y, width, height)
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected.append((label, confidence))

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    final_detected = []
    if len(indices) > 0:
        for i in indices.flatten():
            label = CLASSES[class_ids[i]]
            conf = confidences[i]
            final_detected.append((label, conf))

    return final_detected


def extract_numbers(image):
    """Use Tesseract OCR to detect and extract numbers from the image with confidence scores"""
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to make text larger (helps with small text)
    height, width = gray.shape
    scale_factor = max(1, 300 // min(height, width))  # Scale up if smaller than 300px
    if scale_factor > 1:
        gray = cv2.resize(
            gray,
            (width * scale_factor, height * scale_factor),
            interpolation=cv2.INTER_CUBIC,
        )

    # Try multiple preprocessing approaches
    preprocessed_images = []

    # Method 1: Original thresholding
    _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh1)

    # Method 2: Adaptive thresholding
    thresh2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    preprocessed_images.append(thresh2)

    # Method 3: Simple binary threshold
    _, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    preprocessed_images.append(thresh3)

    # Method 4: Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh4 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(thresh4)

    # Method 5: Gaussian blur + threshold (helps with noisy images)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh5)

    all_results = []  # List of (number, confidence) tuples
    for i, processed_img in enumerate(preprocessed_images):
        # Convert numpy array to PIL Image for tesserocr
        pil_image = Image.fromarray(processed_img)

        # Try different PSM modes for better detection
        psm_modes = [6, 7, 8]  # 6=uniform block, 7=single line, 8=single word
        for psm in psm_modes:
            # Run Tesseract with digit-only config
            with PyTessBaseAPI() as api:
                api.SetImage(pil_image)
                api.SetVariable("tessedit_char_whitelist", "0123456789")  # Only digits
                api.SetVariable("tessedit_pageseg_mode", str(psm))
                text = api.GetUTF8Text()
                confidence = api.MeanTextConf()

            # Extract all numbers (integers or floats) from the text
            import re

            numbers = re.findall(r"\d+\.?\d*", text.strip())  # Find all numeric strings
            numbers = [float(num) if "." in num else int(num) for num in numbers]

            # Add each number with its confidence score
            for num in numbers:
                all_results.append((num, confidence))

    # Remove duplicates, keeping the highest confidence for each number
    number_conf_map = {}
    for num, conf in all_results:
        if num not in number_conf_map or conf > number_conf_map[num]:
            number_conf_map[num] = conf

    # Filter out numbers below the confidence threshold
    filtered_conf_map = {
        num: conf
        for num, conf in number_conf_map.items()
        if conf >= OCR_CONFIDENCE_THRESHOLD * 100
    }  # Convert to 0-100 scale

    # Sort by confidence (highest first)
    sorted_results = sorted(filtered_conf_map.items(), key=lambda x: x[1], reverse=True)

    # Return just the numbers (for backward compatibility) and the filtered confidence map
    return [num for num, conf in sorted_results], filtered_conf_map


def analyzeExistingImages(output_dir="./captured_images", target_objects=None):
    """Analyze all images in captured_images folder"""
    if not os.path.exists(output_dir):
        print("Error: Directory {} not found. Capture images first.".format(output_dir))
        return

    net = load_detection_model()
    if not net:
        return

    image_files = [
        f
        for f in os.listdir(output_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        print("No images found in {}. Capture some first.".format(output_dir))
        return

    print(
        "Analyzing {} images with YOLOv3-Tiny (COCO 80 classes) + Number OCR...".format(
            len(image_files)
        )
    )
    if target_objects:
        print("Filtering for: {}".format(", ".join(target_objects)))
    else:
        print("Detecting all objects.")

    for filename in sorted(image_files):
        filepath = os.path.join(output_dir, filename)
        print("\nProcessing: {}".format(filename))

        image = cv2.imread(filepath)
        if image is None:
            print("  Could not load image.")
            continue

        t0 = time.time()
        detected = detect_objects(image, net, target_objects)
        t1 = time.time()

        print("  Inference time: {:.3f} seconds".format(t1 - t0))

        if detected:
            print("  Detected objects:")
            for label, conf in detected:
                print("    - {} ({:.1f}%)".format(label, conf * 100))

            # Print the object with the highest confidence
            max_obj = max(detected, key=lambda x: x[1])
            print(
                "  Highest confidence object: {} ({:.1f}%)".format(
                    max_obj[0], max_obj[1] * 100
                )
            )
        else:
            print("  No objects above threshold detected.")

        # Additional: Run OCR for numbers
        numbers, confidence_map = extract_numbers(image)
        if numbers:
            print(
                "  Detected numbers via OCR (confidence >= {:.0f}%):".format(
                    OCR_CONFIDENCE_THRESHOLD * 100
                )
            )
            print("    - {}".format(numbers))
            # Print the number with the highest confidence
            if confidence_map:
                max_num = max(confidence_map.items(), key=lambda x: x[1])
                print(
                    "  Highest confidence number: {} ({:.1f}%)".format(
                        max_num[0], max_num[1]
                    )
                )
        else:
            print(
                "  No numbers detected via OCR (confidence >= {:.0f}%).".format(
                    OCR_CONFIDENCE_THRESHOLD * 100
                )
            )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    print("Vision Offline Test – YOLOv3-Tiny (COCO 80 classes) + Number OCR")
    print("=" * 60)

    # Example: Detect only person and bottle
    # Set to None to detect all 80 classes
    target_objects = ["person", "bottle", "sports ball", "cell phone", "book"]
    # target_objects = None  # Uncomment for all objects

    analyzeExistingImages("./test_images", target_objects=target_objects)
