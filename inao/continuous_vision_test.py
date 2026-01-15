# -*- encoding: UTF-8 -*-
# Continuous Vision Processing with Visualization: Fetch frames continuously from NAO, display live video, process with YOLO + OCR
# Builds on vision_works.py (image capture) + vision_v3_test.py (detection)
# Processes frames in real-time on laptop with live video display

import sys
import time
import cv2
import numpy as np
import os
from tesserocr import PyTessBaseAPI  # For OCR (Python 2.7 compatible)
from PIL import Image  # For image conversion to OCR

from naoqi import ALProxy

sys.path.insert(0, "/home/georg/Desktop/hands_on_nao/inao")
from inao import NAO

# Model setup – place files in ./models/
MODEL_DIR = "./models"
CFG_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
NAMES_FILE = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO objects
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
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU
    print("YOLOv3-Tiny model loaded successfully (80 COCO classes).")
    return net


def detect_objects(image, net, target_objects=None):
    """Run YOLOv3-Tiny detection and return filtered list"""
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
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
                if target_objects is not None and label not in target_objects:
                    continue

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale_factor = max(1, 300 // min(height, width))
    if scale_factor > 1:
        gray = cv2.resize(
            gray,
            (width * scale_factor, height * scale_factor),
            interpolation=cv2.INTER_CUBIC,
        )

    preprocessed_images = []
    _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh1)
    thresh2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    preprocessed_images.append(thresh2)
    _, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    preprocessed_images.append(thresh3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh4 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(thresh4)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh5)

    all_results = []
    for processed_img in preprocessed_images:
        pil_image = Image.fromarray(processed_img)
        psm_modes = [6, 7, 8]
        for psm in psm_modes:
            with PyTessBaseAPI() as api:
                api.SetImage(pil_image)
                api.SetVariable("tessedit_char_whitelist", "0123456789")
                api.SetVariable("tessedit_pageseg_mode", str(psm))
                text = api.GetUTF8Text()
                confidence = api.MeanTextConf()

            import re

            numbers = re.findall(r"\d+\.?\d*", text.strip())
            numbers = [float(num) if "." in num else int(num) for num in numbers]
            for num in numbers:
                all_results.append((num, confidence))

    number_conf_map = {}
    for num, conf in all_results:
        if num not in number_conf_map or conf > number_conf_map[num]:
            number_conf_map[num] = conf

    filtered_conf_map = {
        num: conf
        for num, conf in number_conf_map.items()
        if conf >= OCR_CONFIDENCE_THRESHOLD * 100
    }
    sorted_results = sorted(filtered_conf_map.items(), key=lambda x: x[1], reverse=True)
    return [num for num, conf in sorted_results], filtered_conf_map


def continuousVisionProcessing(IP, PORT, target_objects=None, max_frames=None):
    """
    Continuously fetch frames from NAO, display live video, process with YOLO + OCR on laptop.
    Runs indefinitely or until max_frames reached. Press 'q' in video window or Ctrl+C to stop.
    """
    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    # Create NAO instance for speech
    nao = NAO(IP)
    resolution = 2  # VGA
    colorSpace = 11  # RGB

    net = load_detection_model()
    if not net:
        return

    # Subscribe to video feed
    videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)
    print(
        "Subscribed to NAO video feed. Processing frames continuously with live visualization..."
    )
    if target_objects:
        print("Filtering for: {}".format(", ".join(target_objects)))
    else:
        print("Detecting all objects.")
    print("Press 'q' in video window or Ctrl+C to stop.")

    frame_count = 0
    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break

            t0 = time.time()
            naoImage = camProxy.getImageRemote(videoClient)
            t1 = time.time()

            print(
                "\nFrame {} - Acquisition delay: {:.3f} seconds".format(
                    frame_count, t1 - t0
                )
            )

            # Process frame
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_array = np.fromstring(array, dtype=np.uint8).reshape(
                (imageHeight, imageWidth, 3)
            )
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # YOLO object detection
            detected = detect_objects(bgr_image, net, target_objects)
            if detected:
                print("  Detected objects:")
                for label, conf in detected:
                    print("    - {} ({:.1f}%)".format(label, conf * 100))
                max_obj = max(detected, key=lambda x: x[1])
                print(
                    "  Highest confidence object: {} ({:.1f}%)".format(
                        max_obj[0], max_obj[1] * 100
                    )
                )
                # Count objects by type
                object_counts = {}
                for label, conf in detected:
                    object_counts[label] = object_counts.get(label, 0) + 1
                # Generate speech string for counts
                speech_parts = []
                for label, count in object_counts.items():
                    if count == 1:
                        speech_parts.append("a {}".format(label))
                    else:
                        speech_parts.append("{} {}s".format(count, label))
                speech_text = "I see " + " and ".join(speech_parts)
                print("  Speech: {}".format(speech_text))
                # Robot speaks the counts
                nao.tts.say(speech_text)
            else:
                print("  No objects above threshold detected.")

            # OCR number detection
            numbers, confidence_map = extract_numbers(bgr_image)
            if numbers:
                print(
                    "  Detected numbers via OCR (confidence >= {:.0f}%): {}".format(
                        OCR_CONFIDENCE_THRESHOLD * 100, numbers
                    )
                )
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

            # Visualize frame in live video window
            cv2.imshow("NAO Live Vision", bgr_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                break

            frame_count += 1
            time.sleep(0.5)  # Brief delay between frames

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print("Error: {}".format(e))
    finally:
        camProxy.unsubscribe(videoClient)
        cv2.destroyAllWindows()
        print("Unsubscribed from video feed. Processed {} frames.".format(frame_count))


if __name__ == "__main__":
    IP = "192.168.1.118"
    PORT = 9559

    if len(sys.argv) > 1:
        IP = sys.argv[1]

    # Example: Detect specific objects
    target_objects = None
    # target_objects = None  # For all objects

    # Run continuously (set max_frames=10 for testing)
    continuousVisionProcessing(IP, PORT, target_objects=target_objects, max_frames=None)
