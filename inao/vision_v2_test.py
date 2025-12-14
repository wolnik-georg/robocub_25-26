# -*- encoding: UTF-8 -*-
# Vision Offline Test with YOLOv3-Tiny (COCO 80 classes)
# Analyzes existing images from ./captured_images
# No NAO connection needed – detection runs on laptop only

import sys
import time
import cv2
import numpy as np
import os

# Model setup – place files in ./models/
MODEL_DIR = "./models"
CFG_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
NAMES_FILE = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to report
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold


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
        "Analyzing {} images with YOLOv3-Tiny (COCO 80 classes)...".format(
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

            # NEW: Print the object with the highest confidence
            max_obj = max(detected, key=lambda x: x[1])
            print(
                "  Highest confidence object: {} ({:.1f}%)".format(
                    max_obj[0], max_obj[1] * 100
                )
            )
        else:
            print("  No objects above threshold detected.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    print("Vision Offline Test – YOLOv3-Tiny (COCO 80 classes)")
    print("=" * 60)

    # Example: Detect only person and bottle
    # Set to None to detect all 80 classes
    target_objects = None
    # target_objects = None  # Uncomment for all objects

    analyzeExistingImages("./test_images", target_objects=target_objects)
