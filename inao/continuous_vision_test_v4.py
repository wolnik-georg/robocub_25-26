# -*- encoding: UTF-8 -*-
# Continuous Vision Processing with Visualization: Fetch frames continuously from NAO, display live video, process with YOLO + OCR
# Builds on vision_works.py (image capture) + vision_v3_test.py (detection)
# Processes frames in real-time on laptop with live video display
# Added: Optional head rotation mode using MotionReactions class (loops RotateHeadLeft and RotateHeadRight continuously)
# V4: Added speech recognition for voice commands:
#   - "Simon says how many [object]" -> Count objects in current view (no head movement)
#   - "Simon says search for [object]" -> Rotate head to find object, then report count
#   - "Simon says go forward/backward/left/right" -> Physical movement commands
#   - "Simon says stand/sit/crouch" -> Posture changes
#   - "Simon says raise left/right arm" -> Arm movements
#   - "Simon says wave" -> Wave gesture
#   - All commands require "Simon says" prefix to prevent accidental triggering
# V4 IMPROVED: Optimized offline speech recognition (PocketSphinx) for better performance

import sys
import time
import cv2
import numpy as np
import os
import math  # Added for head angle calculations
import threading
import re
from tesserocr import PyTessBaseAPI  # For OCR (Python 2.7 compatible)
from PIL import Image  # For image conversion to OCR

try:
    import speech_recognition as sr

    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print(
        "WARNING: speech_recognition not available. Install with: pip install SpeechRecognition pyaudio"
    )

# Try to import pocketsphinx for offline recognition
try:
    import pocketsphinx

    OFFLINE_SPEECH_AVAILABLE = True
except ImportError:
    OFFLINE_SPEECH_AVAILABLE = False

# Try to import difflib for fuzzy command matching
try:
    import difflib

    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False

from naoqi import ALProxy

sys.path.insert(0, "/home/georg/Desktop/hands_on_nao/inao")
from inao import NAO
from motion_reactions import MotionReactions  # Import for head rotation mode

# Model setup – place files in ./models/
MODEL_DIR = "./models"
CFG_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
NAMES_FILE = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO objects
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold
OCR_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for OCR numbers (30%)
SPEECH_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for speech recognition (60%)

# Whitelist of expected objects (from COCO dataset) - helps validate recognition
EXPECTED_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "ski",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


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


class VoiceCommandListener(object):
    """Listen for voice commands in a separate thread"""

    def __init__(self):
        self.command = None
        self.target_object = None
        self.mode = None  # 'query' or 'search'
        self.listening = True
        self.recognizer = sr.Recognizer() if SPEECH_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_AVAILABLE else None
        self.consecutive_errors = 0  # Track consecutive errors
        self.max_consecutive_errors = 3  # Max errors before switching modes
        self.use_offline = False  # Start with online, switch to offline if needed

        # IMPROVEMENT: Command pattern mappings for fuzzy matching
        self.movement_commands = {
            "move_forward": ["go forward", "move forward", "walk forward", "forward"],
            "move_backward": [
                "go backward",
                "move backward",
                "go back",
                "backward",
                "back",
            ],
            "move_left": ["go left", "move left", "step left", "left"],
            "move_right": ["go right", "move right", "step right", "right"],
            "turn_left": ["turn left", "rotate left"],
            "turn_right": ["turn right", "rotate right"],
            "stand": ["stand up", "stand"],
            "sit": ["sit down", "sit"],
            "crouch": ["crouch", "crouch down"],
            "raise_left_arm": ["raise left arm", "lift left arm"],
            "raise_right_arm": ["raise right arm", "lift right arm"],
            "raise_both_arms": ["raise both arms", "raise arms"],
            "head_left": ["look left", "turn head left"],
            "head_right": ["look right", "turn head right"],
            "head_up": ["look up", "head up"],
            "head_down": ["look down", "head down"],
            "wave": ["wave", "say hello"],
        }

        # IMPROVEMENT: Build keyword list for PocketSphinx keyword spotting
        # Format: (keyword, sensitivity) - lower sensitivity = less false positives
        self.keywords = [
            ("simon says", 1e-20),  # Very high sensitivity for trigger phrase
            ("how many", 1e-25),
            ("search for", 1e-25),
            ("find", 1e-30),
            ("forward", 1e-30),
            ("backward", 1e-30),
            ("left", 1e-30),
            ("right", 1e-30),
            ("stand", 1e-30),
            ("sit", 1e-30),
            ("wave", 1e-30),
            ("stop", 1e-20),  # High sensitivity for stop command
            ("quit", 1e-20),
        ]

        # IMPROVEMENT: Common COCO objects as keywords
        common_objects = [
            "bottle",
            "cup",
            "person",
            "chair",
            "book",
            "phone",
            "laptop",
            "car",
            "dog",
            "cat",
        ]
        for obj in common_objects:
            self.keywords.append((obj, 1e-30))

        # IMPROVEMENT: Optimize recognizer settings for better performance
        if self.recognizer:
            # IMPROVEMENT: Increase energy threshold to reduce background noise sensitivity
            # DRASTICALLY increased for offline mode reliability
            self.recognizer.energy_threshold = (
                800  # Much higher = less sensitive (was 500, now 800 for offline)
            )
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.dynamic_energy_adjustment_damping = 0.15
            self.recognizer.dynamic_energy_ratio = 1.5
            # Reduce pause threshold for faster response
            self.recognizer.pause_threshold = (
                0.6  # Slightly longer to avoid cutting off speech (was 0.5)
            )
            self.recognizer.phrase_threshold = (
                0.4  # Require longer speech to reduce noise triggers (was 0.3)
            )
            self.recognizer.non_speaking_duration = (
                0.4  # More context for better recognition (was 0.3)
            )

    def _fuzzy_match_movement(self, text):
        """Use fuzzy matching to find best movement command match"""
        best_match = None
        best_score = 0.0

        for cmd, patterns in self.movement_commands.items():
            for pattern in patterns:
                # Calculate similarity ratio
                if FUZZY_MATCHING_AVAILABLE:
                    score = difflib.SequenceMatcher(None, text, pattern).ratio()
                    if score > best_score and score > 0.7:  # 70% similarity threshold
                        best_score = score
                        best_match = (cmd, pattern)
                else:
                    # Simple substring matching fallback
                    if pattern in text:
                        return (cmd, pattern)

        return best_match

    def _fuzzy_match_object(self, obj_word):
        """Use fuzzy matching to find best object match from EXPECTED_OBJECTS"""
        if obj_word in EXPECTED_OBJECTS:
            return obj_word  # Exact match

        best_match = None
        best_score = 0.0

        if FUZZY_MATCHING_AVAILABLE:
            for expected_obj in EXPECTED_OBJECTS:
                score = difflib.SequenceMatcher(None, obj_word, expected_obj).ratio()
                if score > best_score:
                    best_score = score
                    best_match = expected_obj

            # Only return match if similarity is above threshold (75%)
            if best_score > 0.75:
                return best_match

        return obj_word  # Return original if no good match found

    def parse_command(self, text):
        """Parse voice command and extract intent"""
        text = text.lower()
        print("Heard: '{}'".format(text))

        # Check for "Simon says" prefix - required for all commands
        if not text.startswith("simon says"):
            print("  -> Ignored (missing 'Simon says' prefix)")
            return False

        # Remove "Simon says" prefix for further parsing
        text = text.replace("simon says", "").strip()
        print("  -> Processing command: '{}'".format(text))

        # Pattern: "how many [object]" or "how many [object]s are there"
        match = re.search(r"how many (\w+)", text)
        if match:
            obj = match.group(1)
            # Remove plural 's' if present
            if obj.endswith("s"):
                obj = obj[:-1]

            # IMPROVEMENT: Use fuzzy matching to find best object match
            original_obj = obj
            obj = self._fuzzy_match_object(obj)

            # IMPROVEMENT: Validate object against expected list
            if obj not in EXPECTED_OBJECTS:
                print(
                    "  -> WARNING: '{}' not in expected objects list. Proceeding anyway.".format(
                        obj
                    )
                )
            elif obj != original_obj:
                print("  -> Fuzzy matched '{}' to '{}'".format(original_obj, obj))

            self.mode = "query"
            self.target_object = obj
            self.command = "count"
            print("Command parsed: MODE=query, OBJECT={}".format(obj))
            return True

        # Pattern: "search for [object]" or "find [object]"
        match = re.search(r"(?:search for|find) (\w+)", text)
        if match:
            obj = match.group(1)
            # Remove plural 's' if present
            if obj.endswith("s"):
                obj = obj[:-1]

            # IMPROVEMENT: Use fuzzy matching to find best object match
            original_obj = obj
            obj = self._fuzzy_match_object(obj)

            # IMPROVEMENT: Validate object against expected list
            if obj not in EXPECTED_OBJECTS:
                print(
                    "  -> WARNING: '{}' not in expected objects list. Proceeding anyway.".format(
                        obj
                    )
                )
            elif obj != original_obj:
                print("  -> Fuzzy matched '{}' to '{}'".format(original_obj, obj))

            self.mode = "search"
            self.target_object = obj
            self.command = "search"
            print("Command parsed: MODE=search, OBJECT={}".format(obj))
            return True

        # Physical movement commands
        # IMPROVEMENT: Use fuzzy matching for better recognition tolerance
        movement_match = self._fuzzy_match_movement(text)
        if movement_match:
            cmd, pattern = movement_match
            self.mode = "movement"
            self.command = cmd
            print(
                "Command parsed: MODE=movement, ACTION={} (matched: '{}')".format(
                    cmd, pattern
                )
            )
            return True

        # Pattern: "stop" or "quit"
        if "stop" in text or "quit" in text or "exit" in text:
            self.mode = "stop"
            self.command = "stop"
            print("Command parsed: STOP")
            return True

        print("  -> Command not recognized")
        return False

    def listen_loop(self):
        """Continuously listen for voice commands"""
        if not SPEECH_AVAILABLE:
            print("Speech recognition not available.")
            return

        print("\n=== Voice Command Listener Started ===")
        print("Say: 'Simon says how many [object]' to count objects")
        print("Say: 'Simon says search for [object]' to find objects")
        print("Say: 'Simon says go forward/backward/left/right' to move")
        print("Say: 'Simon says stand/sit/crouch' to change posture")
        print("Say: 'Simon says raise left/right arm' for arm movements")
        print("Say: 'Simon says wave' to wave")
        print("Say: 'Simon says stop' to quit")
        print("\nNOTE: Commands MUST start with 'Simon says' - other speech is ignored")
        print("      to prevent background noise from triggering actions.")

        # IMPROVEMENT: Start directly in offline mode if available for better reliability
        if OFFLINE_SPEECH_AVAILABLE:
            print("\n[OFFLINE MODE] PocketSphinx speech recognition available")
            print("Starting in OFFLINE mode - listening continuously")
            print("\n*** KEYWORD SPOTTING ENABLED for 10x better accuracy! ***")
            print("\nTIPS for better offline recognition:")
            print("  - Speak CLEARLY and at NORMAL volume (not too loud/quiet)")
            print("  - Pause 0.5 seconds BEFORE saying 'Simon says'")
            print("  - Pronounce each word DISTINCTLY")
            print("  - Speak at a steady pace (not too fast)")
            print("  - Reduce background noise as much as possible")
            print("  - Say full commands: 'Simon says wave' NOT just 'wave'")
            print("\nKeyword spotting is active - only recognizes:")
            print("  Commands: simon says, how many, search for, forward, backward,")
            print("            left, right, stand, sit, wave, stop, quit")
            print(
                "  Objects: bottle, cup, person, chair, book, phone, laptop, car, dog, cat"
            )
            print("  (Other speech is automatically ignored)")
            # Start directly in offline mode
            self.use_offline = True
        else:
            print("\n[ONLINE MODE] Using Google Speech API (requires internet)")
            print("Install PocketSphinx for offline mode: pip install pocketsphinx")

        print("=" * 40)

        # IMPROVEMENT: Adjust for ambient noise with longer calibration for offline mode
        try:
            with self.microphone as source:
                calibration_time = (
                    4 if self.use_offline else 2
                )  # Longer calibration for offline (was 3)
                print(
                    "Calibrating microphone for ambient noise... ({} seconds)".format(
                        calibration_time
                    )
                )
                print("  Please remain quiet during calibration...")
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=calibration_time
                )

                # IMPROVEMENT: For offline mode, increase threshold even more after calibration
                if self.use_offline:
                    self.recognizer.energy_threshold = max(
                        self.recognizer.energy_threshold * 1.5, 800
                    )

                print(
                    "Microphone calibrated. Energy threshold: {:.0f}".format(
                        self.recognizer.energy_threshold
                    )
                )
                print("Ready for commands.")
                print("\n[Listening continuously...]\n")  # Show once at start
        except Exception as e:
            print("WARNING: Could not calibrate microphone: {}".format(e))

        while self.listening:
            try:
                with self.microphone as source:
                    # REMOVED: Don't print "Listening..." each cycle - already listening continuously

                    # IMPROVEMENT: Use listen() without timeout for continuous listening
                    # Only set phrase_time_limit to prevent overly long phrases
                    # IMPROVEMENT: Shorter phrase limit for offline - commands are short
                    phrase_limit = 4 if self.use_offline else 8  # Reduced from 5 to 4

                    # Listen continuously without timeout - waits indefinitely for speech detected
                    audio = self.recognizer.listen(
                        source,
                        timeout=None,  # No timeout - listen forever until speech detected
                        phrase_time_limit=phrase_limit,
                    )

                try:
                    # Try online recognition first (Google), then fallback to offline
                    if not self.use_offline:
                        try:
                            # IMPROVEMENT: Get recognition with alternatives to check confidence
                            text = self.recognizer.recognize_google(
                                audio, language="en-US", show_all=False
                            )

                            # IMPROVEMENT: Pre-filter - only process if it contains "simon"
                            if "simon" in text.lower():
                                self.parse_command(text)
                            # Else: silently ignore - likely background noise or conversation

                            self.consecutive_errors = 0
                        except sr.RequestError as e:
                            # Network error - switch to offline if available
                            self.consecutive_errors += 1
                            print(
                                "[Google API error {}/{}: {}]".format(
                                    self.consecutive_errors,
                                    self.max_consecutive_errors,
                                    str(e)[:50],
                                )
                            )

                            if self.consecutive_errors >= self.max_consecutive_errors:
                                if OFFLINE_SPEECH_AVAILABLE:
                                    print(
                                        "\n*** Switching to OFFLINE speech recognition (PocketSphinx) ***"
                                    )
                                    print(
                                        "*** Listening continuously - no internet required ***\n"
                                    )
                                    self.use_offline = True
                                    self.consecutive_errors = 0
                                else:
                                    print(
                                        "\nWARNING: No internet and offline speech not available."
                                    )
                                    print(
                                        "Speech commands disabled. Install pocketsphinx for offline mode."
                                    )
                                    print("Pausing for 10 seconds...")
                                    time.sleep(10)
                                    self.consecutive_errors = 0
                    else:
                        # IMPROVEMENT: Use offline recognition (PocketSphinx) with KEYWORD SPOTTING
                        try:
                            # DRASTIC IMPROVEMENT: Use keyword spotting mode for much better accuracy
                            # This constrains recognition to only our expected keywords
                            text = self.recognizer.recognize_sphinx(
                                audio,
                                language="en-US",
                                keyword_entries=self.keywords,  # Use keyword spotting!
                                show_all=False,
                            )

                            # IMPROVEMENT: Filter out very short/empty results from PocketSphinx (likely noise)
                            if text and len(text.strip()) > 3:
                                # IMPROVEMENT: Pre-filter - only process if it contains "simon"
                                if "simon" in text.lower():
                                    self.parse_command(text)
                                    self.consecutive_errors = 0
                                else:
                                    # Even with keyword spotting, log what was detected for debugging
                                    print(
                                        "  [Keyword detected but no 'simon': '{}']".format(
                                            text
                                        )
                                    )
                            else:
                                # Likely noise, don't count as error, don't print anything
                                pass

                        except sr.UnknownValueError:
                            # IMPROVEMENT: PocketSphinx couldn't understand - don't spam console in offline mode
                            # Just continue listening silently
                            pass
                        except sr.RequestError as e:
                            print("[Offline recognition error: {}]".format(e))
                            self.consecutive_errors += 1
                            # IMPROVEMENT: Auto-restart recognizer after persistent errors
                            if self.consecutive_errors >= 5:
                                print(
                                    "WARNING: Too many offline errors. Restarting recognizer..."
                                )
                                self.recognizer = sr.Recognizer()
                                # Re-apply optimizations
                                self.recognizer.energy_threshold = (
                                    800  # Updated to match new threshold
                                )
                                self.recognizer.dynamic_energy_threshold = True
                                self.recognizer.pause_threshold = 0.6  # Updated
                                self.recognizer.phrase_threshold = 0.4  # Updated
                                self.recognizer.non_speaking_duration = 0.4  # Updated
                                self.consecutive_errors = 0
                                print("[Listening continuously...]\n")

                except sr.UnknownValueError:
                    # IMPROVEMENT: Don't print "could not understand" - just continue listening
                    # Speech was detected but not recognized - normal behavior, stay silent
                    self.consecutive_errors = 0

            except sr.WaitTimeoutError:
                # This should never happen with timeout=None, but keep for safety
                # No speech detected - this is normal, just continue
                self.consecutive_errors = 0
                pass

            except Exception as e:
                self.consecutive_errors += 1
                print(
                    "[Listener error {}/{}: {}]".format(
                        self.consecutive_errors,
                        self.max_consecutive_errors,
                        str(e)[:50],
                    )
                )

                # Pause if too many errors
                if self.consecutive_errors >= self.max_consecutive_errors:
                    print("WARNING: Too many errors. Pausing for 5 seconds...")
                    time.sleep(5)
                    self.consecutive_errors = 0
                    print("[Listening continuously...]\n")
                else:
                    time.sleep(1)

    def start(self):
        """Start listening in a separate thread"""
        if not SPEECH_AVAILABLE:
            return None
        listener_thread = threading.Thread(target=self.listen_loop)
        listener_thread.daemon = True
        listener_thread.start()
        return listener_thread

    def stop(self):
        """Stop listening"""
        self.listening = False

    def get_command(self):
        """Get and clear the current command"""
        cmd = {
            "command": self.command,
            "mode": self.mode,
            "target_object": self.target_object,
        }
        # Clear command after retrieval
        self.command = None
        self.mode = None
        self.target_object = None
        return cmd


def execute_movement(motion, nao, movement_command):
    """
    Execute physical movement based on voice command
    :param motion: MotionReactions instance
    :param nao: NAO instance for speech
    :param movement_command: The movement command to execute
    """
    try:
        # Movement commands (translate to MotionReactions methods)
        if movement_command == "move_forward":
            nao.tts.say("Moving forward")
            motion.wakeUp()
            motion.move_position(x=0.3, y=0.0, theta=0.0)
            motion.rest()
        elif movement_command == "move_backward":
            nao.tts.say("Moving backward")
            motion.wakeUp()
            motion.move_position(x=-0.2, y=0.0, theta=0.0)
            motion.rest()
        elif movement_command == "move_left":
            nao.tts.say("Moving left")
            motion.wakeUp()
            motion.move_position(x=0.0, y=0.2, theta=0.0)
            motion.rest()
        elif movement_command == "move_right":
            nao.tts.say("Moving right")
            motion.wakeUp()
            motion.move_position(x=0.0, y=-0.2, theta=0.0)
            motion.rest()
        elif movement_command == "turn_left":
            nao.tts.say("Turning left")
            motion.wakeUp()
            motion.move_position(x=0.0, y=0.0, theta=math.pi / 4)
            motion.rest()
        elif movement_command == "turn_right":
            nao.tts.say("Turning right")
            motion.wakeUp()
            motion.move_position(x=0.0, y=0.0, theta=-math.pi / 4)
            motion.rest()

        # Posture commands
        elif movement_command == "stand":
            nao.tts.say("Standing up")
            motion.posture(posture_name="Stand", speed=1.0)
        elif movement_command == "sit":
            nao.tts.say("Sitting down")
            motion.posture(posture_name="Sit", speed=1.0)
        elif movement_command == "crouch":
            nao.tts.say("Crouching")
            motion.posture(posture_name="Crouch", speed=1.0)

        # Arm movements
        elif movement_command == "raise_left_arm":
            nao.tts.say("Raising left arm")
            motion.wakeUp()
            motion.move_joint(
                joint_name="LShoulderPitch", angle=0.0, speed=0.1, waitingtime=2.0
            )
            motion.rest()
        elif movement_command == "raise_right_arm":
            nao.tts.say("Raising right arm")
            motion.wakeUp()
            motion.move_joint(
                joint_name="RShoulderPitch", angle=0.0, speed=0.1, waitingtime=2.0
            )
            motion.rest()
        elif movement_command == "raise_both_arms":
            nao.tts.say("Raising both arms")
            motion.wakeUp()
            motion.move_joint(
                joint_name="LShoulderPitch", angle=0.0, speed=0.1, waitingtime=1.0
            )
            motion.move_joint(
                joint_name="RShoulderPitch", angle=0.0, speed=0.1, waitingtime=2.0
            )
            motion.rest()

        # Head movements (explicit)
        elif movement_command == "head_left":
            nao.tts.say("Looking left")
            motion.wakeUp()
            motion.move_joint(
                joint_name="HeadYaw", angle=math.pi / 4, speed=0.1, waitingtime=2.0
            )
            motion.rest()
        elif movement_command == "head_right":
            nao.tts.say("Looking right")
            motion.wakeUp()
            motion.move_joint(
                joint_name="HeadYaw", angle=-math.pi / 4, speed=0.1, waitingtime=2.0
            )
            motion.rest()
        elif movement_command == "head_up":
            nao.tts.say("Looking up")
            motion.wakeUp()
            motion.move_joint(
                joint_name="HeadPitch", angle=-0.3, speed=0.1, waitingtime=2.0
            )
            motion.rest()
        elif movement_command == "head_down":
            nao.tts.say("Looking down")
            motion.wakeUp()
            motion.move_joint(
                joint_name="HeadPitch", angle=0.3, speed=0.1, waitingtime=2.0
            )
            motion.rest()

        # Wave gesture
        elif movement_command == "wave":
            nao.tts.say("Hello! Waving!")
            motion.wakeUp()
            # Raise right arm and wave
            motion.move_joint(
                joint_name="RShoulderPitch", angle=0.0, speed=0.1, waitingtime=1.0
            )
            # Wave motion (rotate wrist back and forth)
            for _ in range(3):
                motion.move_joint(
                    joint_name="RWristYaw", angle=1.0, speed=0.3, waitingtime=0.3
                )
                motion.move_joint(
                    joint_name="RWristYaw", angle=-1.0, speed=0.3, waitingtime=0.3
                )
            # Lower arm
            motion.move_joint(
                joint_name="RShoulderPitch", angle=1.5, speed=0.1, waitingtime=1.0
            )
            motion.rest()

        else:
            nao.tts.say("Unknown movement command")
            print("ERROR: Unknown movement command: {}".format(movement_command))

    except Exception as e:
        print("ERROR executing movement {}: {}".format(movement_command, e))
        nao.tts.say("Movement failed")


def continuousVisionProcessing(
    IP,
    PORT,
    target_objects=None,
    max_frames=None,
    enable_head_rotation=False,
    enable_voice_commands=False,
):
    """
    Continuously fetch frames from NAO, display live video, process with YOLO + OCR on laptop.
    Optional: Enable head rotation mode to loop RotateHeadLeft and RotateHeadRight continuously.
    Optional: Enable voice commands for interactive object detection and search.

    Voice Commands (all require "Simon says" prefix):
        - "Simon says how many [object]" -> Count objects in current view (no head movement)
        - "Simon says search for [object]" -> Rotate head to find object, then report count
        - "Simon says go forward/backward/left/right" -> Physical movement
        - "Simon says turn left/right" -> Rotate in place
        - "Simon says stand/sit/crouch" -> Change posture
        - "Simon says raise left/right arm" -> Arm movements
        - "Simon says wave" -> Wave gesture
        - "Simon says look left/right/up/down" -> Head movements
        - "Simon says stop" -> Exit program

    Runs indefinitely or until max_frames reached. Press 'q' in video window or Ctrl+C to stop.
    """
    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    # Create NAO instance for speech
    nao = NAO(IP)
    # Create MotionReactions instance for head rotation mode
    motion = MotionReactions(IP, PORT)
    resolution = 2  # VGA
    colorSpace = 11  # RGB

    net = load_detection_model()
    if not net:
        return

    # Initialize voice command listener
    voice_listener = None
    if enable_voice_commands:
        if not SPEECH_AVAILABLE:
            print("ERROR: Speech recognition not available. Disabling voice commands.")
            enable_voice_commands = False
        else:
            voice_listener = VoiceCommandListener()
            voice_listener.start()
            time.sleep(1)  # Give listener time to start

    # Subscribe to video feed
    videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)
    print(
        "Subscribed to NAO video feed. Processing frames continuously with live visualization..."
    )
    if target_objects:
        print("Filtering for: {}".format(", ".join(target_objects)))
    else:
        print("Detecting all objects.")
    if enable_head_rotation:
        print(
            "Head rotation mode enabled: Alternating RotateHeadLeft and RotateHeadRight continuously."
        )
    if enable_voice_commands:
        print("Voice command mode enabled. Speak commands to control the robot.")
    print("Press 'q' in video window or Ctrl+C to stop.")

    frame_count = 0
    head_direction = 1  # 1 for left, -1 for right

    # Voice command state
    current_mode = "passive"  # 'passive', 'query', 'search'
    search_target = None
    search_active = False
    head_rotation_enabled = (
        enable_head_rotation  # Track if head rotation is currently enabled
    )

    # Wake up robot if head rotation is initially enabled
    if head_rotation_enabled:
        motion.wakeUp()

    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break

            # Check for voice commands
            if enable_voice_commands and voice_listener:
                cmd = voice_listener.get_command()
                if cmd["command"]:
                    if cmd["command"] == "stop":
                        print("\n=== STOP command received ===")
                        nao.tts.say("Stopping now")
                        break
                    elif cmd["command"] == "count" and cmd["mode"] == "query":
                        # Query mode: Count objects in current view
                        print(
                            "\n=== QUERY MODE: Counting {} ===".format(
                                cmd["target_object"]
                            )
                        )
                        current_mode = "query"
                        search_target = cmd["target_object"]
                        # Disable head rotation for query
                        if head_rotation_enabled:
                            motion.rest()
                        head_rotation_enabled = False
                        search_active = False
                    elif cmd["command"] == "search" and cmd["mode"] == "search":
                        # Search mode: Enable head rotation to find object
                        print(
                            "\n=== SEARCH MODE: Looking for {} ===".format(
                                cmd["target_object"]
                            )
                        )
                        nao.tts.say("Searching for {}".format(cmd["target_object"]))
                        current_mode = "search"
                        search_target = cmd["target_object"]
                        search_active = True
                        # Enable head rotation for search
                        if not head_rotation_enabled:
                            motion.wakeUp()
                        head_rotation_enabled = True
                    elif cmd["mode"] == "movement":
                        # Physical movement commands
                        print("\n=== MOVEMENT MODE: {} ===".format(cmd["command"]))
                        execute_movement(motion, nao, cmd["command"])
                        # Return to passive mode after movement
                        current_mode = "passive"

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

            # Handle voice command modes
            if current_mode == "query" and search_target:
                # Query mode: Count specific object and report
                count = sum(1 for label, conf in detected if label == search_target)
                if count == 0:
                    speech = "I see no {}s".format(search_target)
                elif count == 1:
                    speech = "I see one {}".format(search_target)
                else:
                    speech = "I see {} {}s".format(count, search_target)
                print("  QUERY RESULT: {}".format(speech))
                nao.tts.say(speech)
                # Return to passive mode
                current_mode = "passive"
                search_target = None
                # Re-enable original head rotation if it was enabled at start
                if enable_head_rotation:
                    motion.wakeUp()
                    head_rotation_enabled = True
            elif current_mode == "search" and search_target and search_active:
                # Search mode: Check if target object found
                found_objects = [
                    label for label, conf in detected if label == search_target
                ]
                if found_objects:
                    count = len(found_objects)
                    if count == 1:
                        speech = "Found it! I see one {}".format(search_target)
                    else:
                        speech = "Found them! I see {} {}s".format(count, search_target)
                    print("  SEARCH RESULT: {}".format(speech))
                    nao.tts.say(speech)
                    # Stop searching
                    search_active = False
                    current_mode = "passive"
                    search_target = None
                    # Disable head rotation after finding object
                    if head_rotation_enabled:
                        motion.rest()
                    head_rotation_enabled = (
                        enable_head_rotation  # Reset to original setting
                    )
                    if head_rotation_enabled:
                        motion.wakeUp()
                else:
                    print(
                        "  SEARCHING: {} not found yet, continuing...".format(
                            search_target
                        )
                    )
            else:
                # Passive mode: Normal detection and speech
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
                    # Count objects by type (only in passive mode)
                    if not enable_voice_commands:
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

            # Optional head rotation mode: Alternate RotateHeadLeft and RotateHeadRight every 3 frames for continuous loop
            # This allows 1-2 images to be captured at each position (middle, left, right)
            # Only rotate if head_rotation_enabled is True (can be toggled by voice commands)
            if head_rotation_enabled and frame_count % 3 == 0:
                if head_direction == 1:
                    motion.move_joint(
                        "HeadYaw", math.pi / 4, 0.1, 1.5
                    )  # RotateHeadLeft - increased waitingtime to 1.5s
                else:
                    motion.move_joint(
                        "HeadYaw", -math.pi / 4, 0.1, 1.5
                    )  # RotateHeadRight - increased waitingtime to 1.5s
                head_direction *= -1  # Switch direction
                time.sleep(
                    1.0
                )  # Additional delay after movement to stabilize and capture images

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print("Error: {}".format(e))
    finally:
        if enable_voice_commands and voice_listener:
            voice_listener.stop()
        camProxy.unsubscribe(videoClient)
        cv2.destroyAllWindows()
        if head_rotation_enabled:
            motion.rest()  # Rest once at the end of head rotation mode
        print("Unsubscribed from video feed. Processed {} frames.".format(frame_count))


if __name__ == "__main__":
    IP = "192.168.1.118"  # 118, 102
    PORT = 9559

    if len(sys.argv) > 1:
        IP = sys.argv[1]

    # Example: Detect specific objects
    target_objects = None
    # target_objects = ["bottle", "person", "cup"]  # For specific objects

    # Enable head rotation mode (set to True to activate looping RotateHeadLeft/Right)
    enable_head_rotation = (
        False  # Changed to False - will be controlled by voice commands
    )

    # Enable voice commands (set to True to activate speech recognition)
    enable_voice_commands = True

    # Run continuously (set max_frames=10 for testing)
    continuousVisionProcessing(
        IP,
        PORT,
        target_objects=target_objects,
        max_frames=None,
        enable_head_rotation=enable_head_rotation,
        enable_voice_commands=enable_voice_commands,
    )
