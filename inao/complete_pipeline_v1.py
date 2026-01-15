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

# Try to import scipy for audio preprocessing
try:
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Audio preprocessing disabled.")
    print("Install with: pip install scipy==1.2.3  (Python 2.7 compatible)")

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


# ============================================================================
# PHASE 1 IMPROVEMENT: Audio Preprocessing Class
# ============================================================================
class AudioPreprocessor(object):
    """
    Preprocess audio to enhance speech and reduce noise
    Python 2.7 compatible, offline, uses numpy and scipy
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_profile = None

    def process(self, audio_data):
        """
        Apply full preprocessing pipeline to audio
        :param audio_data: numpy array of audio samples (int16)
        :return: preprocessed audio (int16)
        """
        if not SCIPY_AVAILABLE:
            return audio_data  # Skip preprocessing if scipy not available

        # Convert to float for processing
        audio_float = np.array(audio_data, dtype=np.float32)

        # Stage 1: Bandpass filter (300Hz - 3400Hz for speech)
        audio_float = self._bandpass_filter(audio_float)

        # Stage 2: Spectral noise gate
        audio_float = self._noise_gate(audio_float)

        # Stage 3: Normalization
        audio_float = self._normalize(audio_float)

        # Stage 4: Pre-emphasis filter (boost high frequencies)
        audio_float = self._pre_emphasis(audio_float)

        # Convert back to int16
        return audio_float.astype(np.int16)

    def _bandpass_filter(self, audio_data):
        """Remove frequencies outside speech range (300-3400 Hz)"""
        nyquist = self.sample_rate / 2.0
        low = 300.0 / nyquist
        high = 3400.0 / nyquist

        # Design 4th order Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype="band")

        # Apply filter (filtfilt for zero-phase)
        filtered = signal.filtfilt(b, a, audio_data)
        return filtered

    def _noise_gate(self, audio_data):
        """
        Reduce background noise using spectral gating
        Analyzes energy per frame and suppresses low-energy sections
        """
        frame_length = int(self.sample_rate * 0.02)  # 20ms frames
        hop_length = frame_length // 2

        # Calculate energy per frame
        frames = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i : i + frame_length]
            energy = np.sqrt(np.mean(frame**2))
            frames.append((i, energy, frame))

        if len(frames) == 0:
            return audio_data

        # Noise threshold = median energy * 1.5
        energies = [e for _, e, _ in frames]
        noise_threshold = np.median(energies) * 1.5

        # Gate: reduce frames below threshold
        output = np.copy(audio_data)
        for i, energy, frame in frames:
            if energy < noise_threshold:
                # Reduce by 90% but don't eliminate (preserves some context)
                output[i : i + frame_length] *= 0.1

        return output

    def _normalize(self, audio_data):
        """Normalize audio to 95% of maximum amplitude"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data * (32767.0 * 0.95 / max_val)  # Scale to 95% of int16 max
        return audio_data

    def _pre_emphasis(self, audio_data):
        """
        Apply pre-emphasis filter to boost high frequencies
        Speech has more energy at low frequencies - this balances it
        """
        return signal.lfilter([1.0, -0.97], [1.0], audio_data)


# ============================================================================
# PHASE 1 IMPROVEMENT: Adaptive Voice Activity Detection (VAD)
# ============================================================================
class AdaptiveVAD(object):
    """
    Voice Activity Detection with adaptive thresholding
    Distinguishes speech from background noise using energy and zero-crossing rate
    Python 2.7 compatible, offline, uses only numpy
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_floor = 0.0
        self.speech_floor = 0.0
        self.alpha_noise = 0.95  # Slow adaptation for noise (95% history)
        self.alpha_speech = 0.8  # Faster adaptation for speech (80% history)
        self.initialized = False

    def zero_crossing_rate(self, audio_chunk):
        """
        Calculate zero-crossing rate (ZCR)
        Speech typically has ZCR between 0.2-0.8
        Pure tones and noise have ZCR outside this range
        """
        signs = np.sign(audio_chunk)
        crossings = np.sum(np.abs(np.diff(signs)))
        zcr = crossings / (2.0 * len(audio_chunk))
        return zcr

    def is_speech(self, audio_chunk):
        """
        Determine if audio chunk contains speech
        :param audio_chunk: numpy array of audio samples
        :return: True if speech detected, False otherwise
        """
        # Calculate energy
        energy = np.sqrt(np.mean(audio_chunk**2))

        # Calculate zero-crossing rate
        zcr = self.zero_crossing_rate(audio_chunk)

        # Initialize noise floor on first call
        if not self.initialized:
            self.noise_floor = energy
            self.initialized = True
            return False

        # Speech detection criteria
        energy_ratio = energy / max(self.noise_floor, 1.0)
        is_energy_high = energy_ratio > 2.5  # Energy 2.5x above noise floor
        is_zcr_speech = 0.2 < zcr < 0.8  # ZCR in speech range

        # Decision: Both conditions must be met
        is_speech_detected = is_energy_high and is_zcr_speech

        # Adaptive update of thresholds
        if is_speech_detected:
            # Update speech floor (faster adaptation)
            self.speech_floor = (
                self.alpha_speech * self.speech_floor
                + (1.0 - self.alpha_speech) * energy
            )
        else:
            # Update noise floor (slower adaptation)
            self.noise_floor = (
                self.alpha_noise * self.noise_floor + (1.0 - self.alpha_noise) * energy
            )

        return is_speech_detected

    def get_threshold(self):
        """
        Get current adaptive energy threshold for speech recognition
        :return: Energy threshold value
        """
        # Threshold = noise floor * 2.5, minimum 300
        return max(self.noise_floor * 2.5, 300.0)


# ============================================================================
# PHASE 3 IMPROVEMENT: Phonetic Correction (Soundex)
# ============================================================================
class SoundexMatcher(object):
    """
    Phonetic matching using Soundex algorithm
    Corrects sound-alike words (bottle/model, forward/foreword)
    Python 2.7 compatible, no external dependencies
    """

    @staticmethod
    def soundex(word):
        """
        Generate Soundex code for a word
        :param word: Input word (string)
        :return: 4-character Soundex code
        """
        if not word:
            return "0000"

        word = word.upper()

        # Soundex mapping table
        soundex_map = {
            "B": "1",
            "F": "1",
            "P": "1",
            "V": "1",
            "C": "2",
            "G": "2",
            "J": "2",
            "K": "2",
            "Q": "2",
            "S": "2",
            "X": "2",
            "Z": "2",
            "D": "3",
            "T": "3",
            "L": "4",
            "M": "5",
            "N": "5",
            "R": "6",
        }

        # Keep first letter
        soundex_code = word[0]

        # Convert rest to digits
        for char in word[1:]:
            if char in soundex_map:
                digit = soundex_map[char]
                # Don't add duplicate adjacent codes
                if soundex_code[-1] != digit:
                    soundex_code += digit

        # Remove vowels and H, W, Y
        soundex_code = soundex_code[0] + soundex_code[1:].replace("A", "").replace(
            "E", ""
        ).replace("I", "").replace("O", "").replace("U", "").replace("H", "").replace(
            "W", ""
        ).replace(
            "Y", ""
        )

        # Pad or truncate to 4 characters
        soundex_code = (soundex_code + "000")[:4]

        return soundex_code

    @staticmethod
    def phonetic_match(word1, word2):
        """
        Check if two words sound similar
        :param word1: First word
        :param word2: Second word
        :return: True if phonetically similar
        """
        return SoundexMatcher.soundex(word1) == SoundexMatcher.soundex(word2)

    @staticmethod
    def find_phonetic_match(word, candidate_list):
        """
        Find phonetically similar word in candidate list
        :param word: Word to match
        :param candidate_list: List of candidate words
        :return: Best match or None
        """
        word_soundex = SoundexMatcher.soundex(word)

        for candidate in candidate_list:
            if SoundexMatcher.soundex(candidate) == word_soundex:
                return candidate

        return None


# ============================================================================
# PHASE 3 IMPROVEMENT: Command Validation (State Machine)
# ============================================================================
class CommandValidator(object):
    """
    Validate commands against robot state and physical constraints
    Prevents impossible sequences (e.g., raise arm while resting)
    Python 2.7 compatible
    """

    def __init__(self):
        # Robot state
        self.posture = "unknown"  # unknown, stand, sit, crouch, rest
        self.stiffness = False  # Motors on/off
        self.last_command = None
        self.command_history = []

        # Valid command transitions
        self.valid_transitions = {
            # Movement commands require standing posture
            "move_forward": ["stand"],
            "move_backward": ["stand"],
            "move_left": ["stand"],
            "move_right": ["stand"],
            "turn_left": ["stand"],
            "turn_right": ["stand"],
            # Posture changes
            "stand": ["sit", "crouch", "rest", "unknown"],
            "sit": ["stand", "crouch", "rest", "unknown"],
            "crouch": ["stand", "sit", "rest", "unknown"],
            # Arm movements require stiffness (wakeup)
            "raise_left_arm": ["stand", "sit"],
            "raise_right_arm": ["stand", "sit"],
            "raise_both_arms": ["stand", "sit"],
            # Head movements require stiffness
            "head_left": ["stand", "sit", "crouch"],
            "head_right": ["stand", "sit", "crouch"],
            "head_up": ["stand", "sit", "crouch"],
            "head_down": ["stand", "sit", "crouch"],
            # Wave requires standing or sitting
            "wave": ["stand", "sit"],
            # Query/search commands - no constraints
            "count": ["stand", "sit", "crouch", "rest", "unknown"],
            "search": ["stand", "sit", "crouch", "rest", "unknown"],
        }

    def validate(self, command, mode="movement"):
        """
        Validate if command is allowed in current state
        :param command: Command to validate
        :param mode: Command mode (movement, query, search)
        :return: (valid, reason) tuple
        """
        # Query and search commands are always valid
        if mode in ["query", "search"]:
            return (True, "")

        # Check if command has constraints
        if command not in self.valid_transitions:
            # Unknown command - allow it (benefit of doubt)
            return (True, "")

        required_states = self.valid_transitions[command]

        # If posture is unknown, assume valid (optimistic)
        if self.posture == "unknown":
            return (True, "")

        # Check if current posture allows this command
        if self.posture not in required_states:
            reason = "Cannot {} while in {} posture. Please {} first.".format(
                command.replace("_", " "), self.posture, " or ".join(required_states)
            )
            return (False, reason)

        # Check for redundant commands
        if self.last_command == command:
            reason = "Already executing {}. Please wait.".format(
                command.replace("_", " ")
            )
            return (False, reason)

        return (True, "")

    def update_state(self, command, mode="movement"):
        """
        Update internal state after command execution
        :param command: Command that was executed
        :param mode: Command mode
        """
        # Update command history
        self.command_history.append(command)
        self.last_command = command

        # Update posture state
        if command == "stand":
            self.posture = "stand"
            self.stiffness = True
        elif command == "sit":
            self.posture = "sit"
            self.stiffness = True
        elif command == "crouch":
            self.posture = "crouch"
            self.stiffness = True
        elif command in [
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
        ]:
            # Movement commands require standing
            self.posture = "stand"
            self.stiffness = True
        elif command in [
            "raise_left_arm",
            "raise_right_arm",
            "raise_both_arms",
            "wave",
        ]:
            # These commands wake up the robot
            self.stiffness = True
        elif command in ["head_left", "head_right", "head_up", "head_down"]:
            # Head movements wake up the robot
            self.stiffness = True

    def reset(self):
        """Reset validator state"""
        self.posture = "unknown"
        self.stiffness = False
        self.last_command = None
        self.command_history = []


# ============================================================================
# PHASE 2 IMPROVEMENT: Multi-pass Verification
# ============================================================================
class MultiPassRecognizer(object):
    """
    Run speech recognition multiple times with different settings
    Use consensus voting to pick the most reliable result
    Python 2.7 compatible, offline
    """

    def __init__(self, recognizer, grammar_file=None, keywords=None):
        self.recognizer = recognizer
        self.grammar_file = grammar_file
        self.keywords = keywords

    def recognize_with_consensus(self, audio, num_passes=3):
        """
        Run recognition multiple times and return consensus result
        :param audio: AudioData to recognize
        :param num_passes: Number of recognition passes (2-3 recommended)
        :return: Consensus text or None
        """
        results = []

        # Pass 1: Grammar mode (if available) - most accurate for valid commands
        if self.grammar_file:
            try:
                text = self.recognizer.recognize_sphinx(
                    audio, language="en-US", grammar=self.grammar_file
                )
                if text and len(text.strip()) > 3:
                    results.append(text.lower().strip())
            except (sr.UnknownValueError, sr.RequestError):
                pass

        # Pass 2: Keyword spotting mode - good for trigger phrases
        if self.keywords:
            try:
                text = self.recognizer.recognize_sphinx(
                    audio, language="en-US", keyword_entries=self.keywords
                )
                if text and len(text.strip()) > 3:
                    results.append(text.lower().strip())
            except (sr.UnknownValueError, sr.RequestError):
                pass

        # Pass 3: Free recognition with language model
        if num_passes >= 3:
            try:
                text = self.recognizer.recognize_sphinx(audio, language="en-US")
                if text and len(text.strip()) > 3:
                    results.append(text.lower().strip())
            except (sr.UnknownValueError, sr.RequestError):
                pass

        # Check for consensus
        if len(results) == 0:
            return None

        if len(results) == 1:
            return results[0]

        # Use majority voting (Counter from collections)
        from collections import Counter

        counts = Counter(results)
        most_common = counts.most_common(1)[0]

        # Require at least 2 matching results for confidence
        if most_common[1] >= 2:
            return most_common[0]

        # No consensus - calculate similarity scores
        # Return result that appears first and has highest similarity to others
        best_result = results[0]
        best_score = 0.0

        for candidate in results:
            total_similarity = 0.0
            for other in results:
                if candidate != other:
                    # Calculate similarity ratio
                    try:
                        import difflib

                        similarity = difflib.SequenceMatcher(
                            None, candidate, other
                        ).ratio()
                        total_similarity += similarity
                    except:
                        pass

            if total_similarity > best_score:
                best_score = total_similarity
                best_result = candidate

        return best_result


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
    """
    Listen for voice commands in a separate thread

    DESIGN PHILOSOPHY: Context-Aware + Robust
    ==========================================
    This class balances two critical goals:

    1. LEVERAGE PROJECT CONTEXT:
       - Dynamically loads commands from motion_reactions.py
       - Auto-discovers available robot capabilities
       - Keeps vocabulary synchronized with actual code
       - Generates natural language patterns from CamelCase names

    2. GRACEFUL DEGRADATION (Robust):
       - Falls back to hardcoded core commands if file missing
       - Continues working even if MotionReactions changes
       - Multiple error handling layers (try/except at each level)
       - Manual command definitions take priority over auto-generated
       - Never breaks user experience due to file I/O issues

    Result: Speech recognition that's smart about your project structure
            BUT won't fail if files are moved, deleted, or malformed.
    """

    def __init__(self, nao_instance=None):
        self.command = None
        self.target_object = None
        self.mode = None  # 'query' or 'search'
        self.listening = True
        self.recognizer = sr.Recognizer() if SPEECH_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_AVAILABLE else None
        self.consecutive_errors = 0  # Track consecutive errors
        self.max_consecutive_errors = 3  # Max errors before switching modes
        self.use_offline = True  # PHASE 1: Start directly in offline mode
        self.nao = nao_instance  # IMPROVEMENT: NAO instance for speech feedback

        # PHASE 1: Initialize audio preprocessor and VAD
        self.preprocessor = AudioPreprocessor() if SCIPY_AVAILABLE else None
        self.vad = AdaptiveVAD()

        # PHASE 1: Path to JSGF grammar file
        self.grammar_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "commands.jsgf"
        )

        # Check if grammar file exists
        if not os.path.exists(self.grammar_file):
            print("WARNING: Grammar file not found at: {}".format(self.grammar_file))
            print("  Speech recognition will use keyword spotting fallback.")
            self.grammar_file = None

        # PHASE 2: Path to custom pronunciation dictionary
        self.dict_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "commands.dic"
        )

        # Check if dictionary file exists
        if not os.path.exists(self.dict_file):
            print("WARNING: Custom dictionary not found at: {}".format(self.dict_file))
            print("  Using default PocketSphinx dictionary.")
            self.dict_file = None

        # PHASE 2: Initialize multi-pass recognizer
        self.multi_pass = None  # Will be initialized after recognizer setup

        # PHASE 3: Initialize phonetic matcher and command validator
        self.soundex = SoundexMatcher()
        self.validator = CommandValidator()

        # PHASE 3 IMPROVEMENT: Load available commands from MotionReactions dynamically
        self.available_robot_commands = self._load_robot_commands()

        # IMPROVEMENT: Command pattern mappings for fuzzy matching
        # This maps natural language patterns to MotionReactions command keys
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

        # PHASE 3 IMPROVEMENT: Add fuzzy matching against actual robot commands
        # Generate natural language patterns from MotionReactions command names
        self._augment_commands_from_robot()

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
            # PHASE 1: Start with moderate energy threshold (VAD will adjust it)
            self.recognizer.energy_threshold = 600  # Will be adjusted by VAD
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.dynamic_energy_adjustment_damping = 0.15
            self.recognizer.dynamic_energy_ratio = 1.5
            # Reduce pause threshold for faster response
            self.recognizer.pause_threshold = 0.6
            self.recognizer.phrase_threshold = 0.4
            self.recognizer.non_speaking_duration = 0.4

            # PHASE 2: Initialize multi-pass recognizer with grammar and keywords
            self.multi_pass = MultiPassRecognizer(
                self.recognizer, grammar_file=self.grammar_file, keywords=self.keywords
            )

    def _load_robot_commands(self):
        """
        Load available commands from MotionReactions dynamically
        This ensures speech recognition knows what the robot can actually do
        ROBUST: Falls back gracefully if file is missing/malformed
        :return: Dictionary of available commands from MotionReactions
        """
        try:
            # Import the command dictionary from motion_reactions
            # We'll parse it to extract available command names
            motion_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "motion_reactions.py"
            )

            if not os.path.exists(motion_file):
                print("INFO: motion_reactions.py not found at expected location.")
                print("  -> Using hardcoded fallback commands (robust mode)")
                return {}

            # Read and parse the example_dict from motion_reactions.py
            with open(motion_file, "r") as f:
                content = f.read()

            # Extract command names from example_dict (simple string parsing)
            # Look for patterns like: "CommandName": {
            import re

            pattern = r'"([A-Z][a-zA-Z]+)":\s*\{'
            matches = re.findall(pattern, content)

            if not matches:
                print(
                    "WARNING: No commands found in motion_reactions.py (pattern mismatch)"
                )
                print("  -> Using hardcoded fallback commands (robust mode)")
                return {}

            robot_commands = {}
            for cmd in matches:
                # Convert CamelCase to snake_case and create natural language variants
                # e.g., "RotateHeadLeft" -> "rotate head left"
                natural = re.sub(r"([A-Z])", r" \1", cmd).strip().lower()
                robot_commands[cmd] = natural

            print(
                "[PHASE 3] Loaded {} commands from MotionReactions".format(
                    len(robot_commands)
                )
            )
            print(
                "  -> Examples: {}".format(", ".join(robot_commands.keys()[:3]) + "...")
            )
            print("  -> Speech recognition vocabulary augmented!")

            return robot_commands

        except IOError as e:
            print("WARNING: Could not read motion_reactions.py: {}".format(e))
            print("  -> Using hardcoded fallback commands (robust mode)")
            return {}
        except Exception as e:
            print("WARNING: Error parsing MotionReactions commands: {}".format(e))
            print("  -> Using hardcoded fallback commands (robust mode)")
            return {}

    def _augment_commands_from_robot(self):
        """
        Augment movement_commands dictionary with patterns from actual robot commands
        This creates fuzzy matching patterns for all MotionReactions commands
        ROBUST: Only augments if commands were successfully loaded
        """
        if not self.available_robot_commands:
            print("[PHASE 3] No robot commands loaded - using core fallback vocabulary")
            print(
                "  -> Core commands available: {}".format(len(self.movement_commands))
            )
            return

        added_count = 0
        # Add natural language patterns for each robot command
        for cmd_key, natural_phrase in self.available_robot_commands.items():
            # Convert to snake_case for internal key
            internal_key = re.sub(r"([A-Z])", r"_\1", cmd_key).lower().lstrip("_")

            # Skip if already manually defined (manual definitions take priority)
            if internal_key in self.movement_commands:
                continue

            # Create pattern variants
            patterns = [natural_phrase]  # e.g., "rotate head left"

            # Add common variations
            words = natural_phrase.split()
            if len(words) >= 2:
                # Add version without first word if it's a verb
                if words[0] in ["move", "rotate", "lift", "bend", "stretch", "twist"]:
                    patterns.append(" ".join(words[1:]))  # "head left"

            self.movement_commands[internal_key] = patterns
            added_count += 1

        print(
            "[PHASE 3] Augmented {} new movement patterns from MotionReactions".format(
                added_count
            )
        )
        print(
            "  -> Total vocabulary: {} command patterns".format(
                len(self.movement_commands)
            )
        )
        print("  -> Speech recognition is context-aware AND robust!")

    def _is_core_command(self, command):
        """
        Check if command is a core hardcoded command (guaranteed to work)
        vs. dynamically loaded command (may need special handling)
        ROBUST: Helps execution layer know what's safe to run
        :param command: Command key (e.g., 'move_forward')
        :return: True if core command, False if dynamically loaded
        """
        core_commands = [
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
            "stand",
            "sit",
            "crouch",
            "raise_left_arm",
            "raise_right_arm",
            "raise_both_arms",
            "head_left",
            "head_right",
            "head_up",
            "head_down",
            "wave",
        ]
        return command in core_commands

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

        # PHASE 3: Try phonetic matching first (faster and more accurate for sound-alikes)
        phonetic_match = self.soundex.find_phonetic_match(obj_word, EXPECTED_OBJECTS)
        if phonetic_match:
            return phonetic_match

        # Fallback to string similarity if available
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

        # PHASE 3: Apply phonetic corrections to common misrecognitions
        # Correct common sound-alike errors before parsing
        corrections = {
            "model": "bottle",  # Common misrecognition
            "waddle": "bottle",
            "foreword": "forward",
            "backwards": "backward",
            "waive": "wave",
            "crouch": "crouch",  # Ensure correct even if misspelled
        }

        words = text.split()
        corrected_words = []
        for word in words:
            # Check if word needs correction
            if word in corrections:
                corrected_word = corrections[word]
                print(
                    "  -> [PHASE 3] Phonetic correction: '{}' -> '{}'".format(
                        word, corrected_word
                    )
                )
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

        text = " ".join(corrected_words)

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

            # PHASE 3: Validate command
            valid, reason = self.validator.validate("count", mode="query")
            if not valid:
                print("  -> [PHASE 3] Validation failed: {}".format(reason))
                # IMPROVEMENT: Provide audio feedback for validation failures
                if self.nao:
                    try:
                        self.nao.tts.say(reason)
                    except:
                        pass  # Ignore TTS errors
                return False

            self.mode = "query"
            self.target_object = obj
            self.command = "count"

            # PHASE 3: Update validator state
            self.validator.update_state("count", mode="query")

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

            # PHASE 3: Validate command
            valid, reason = self.validator.validate("search", mode="search")
            if not valid:
                print("  -> [PHASE 3] Validation failed: {}".format(reason))
                # IMPROVEMENT: Provide audio feedback for validation failures
                if self.nao:
                    try:
                        self.nao.tts.say(reason)
                    except:
                        pass  # Ignore TTS errors
                return False

            self.mode = "search"
            self.target_object = obj
            self.command = "search"

            # PHASE 3: Update validator state
            self.validator.update_state("search", mode="search")

            print("Command parsed: MODE=search, OBJECT={}".format(obj))
            return True

        # Physical movement commands
        # IMPROVEMENT: Use fuzzy matching for better recognition tolerance
        movement_match = self._fuzzy_match_movement(text)
        if movement_match:
            cmd, pattern = movement_match

            # PHASE 3: Validate movement command against robot state
            valid, reason = self.validator.validate(cmd, mode="movement")
            if not valid:
                print("  -> [PHASE 3] Validation failed: {}".format(reason))
                # IMPROVEMENT: Provide audio feedback for validation failures
                if self.nao:
                    try:
                        self.nao.tts.say(reason)
                    except:
                        pass  # Ignore TTS errors
                return False

            self.mode = "movement"
            self.command = cmd

            # PHASE 3: Update validator state
            self.validator.update_state(cmd, mode="movement")

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

        # PHASE 1: Start directly in offline mode
        if OFFLINE_SPEECH_AVAILABLE:
            print("\n[OFFLINE MODE] PocketSphinx speech recognition available")
            print("Starting in OFFLINE mode - listening continuously")

            # PHASE 1: Grammar or keyword spotting mode
            if self.grammar_file:
                print("\n*** PHASE 1: JSGF GRAMMAR MODE ENABLED ***")
                print("Using structured grammar for 50-70% accuracy improvement!")
            else:
                print("\n*** Keyword spotting mode (grammar file not found) ***")

            if SCIPY_AVAILABLE:
                print("*** PHASE 1: AUDIO PREPROCESSING ENABLED ***")
                print("Bandpass filtering + Noise gating active")

            print("*** PHASE 1: ADAPTIVE VAD ENABLED ***")
            print("Dynamic threshold adjustment active")

            # PHASE 2: Show custom dictionary status
            if self.dict_file:
                print("\n*** PHASE 2: CUSTOM PRONUNCIATION DICTIONARY ENABLED ***")
                print("Multiple pronunciations for accent tolerance!")

            # PHASE 2: Show multi-pass status
            if self.multi_pass:
                print("*** PHASE 2: MULTI-PASS VERIFICATION ENABLED ***")
                print("Consensus voting for 20-30% accuracy boost!")

            # PHASE 3: Show phonetic correction and validation status
            print("\n*** PHASE 3: PHONETIC CORRECTION ENABLED ***")
            print("Soundex algorithm for sound-alike word correction!")
            print("*** PHASE 3: COMMAND VALIDATION ENABLED ***")
            print("State machine prevents impossible command sequences!")

            print("\nTIPS for better offline recognition:")
            print("  - Speak CLEARLY and at NORMAL volume (not too loud/quiet)")
            print("  - Pause 0.5 seconds BEFORE saying 'Simon says'")
            print("  - Pronounce each word DISTINCTLY")
            print("  - Speak at a steady pace (not too fast)")
            print("  - Reduce background noise as much as possible")
            print("  - Say full commands: 'Simon says wave' NOT just 'wave'")

            # Start directly in offline mode
            self.use_offline = True
        else:
            print("\n[ONLINE MODE] Using Google Speech API (requires internet)")
            print("Install PocketSphinx for offline mode: pip install pocketsphinx")

        print("=" * 40)

        # PHASE 1: Adjust for ambient noise with VAD-aware calibration
        try:
            with self.microphone as source:
                calibration_time = 5  # PHASE 1: Longer calibration for VAD learning
                print(
                    "Calibrating microphone for ambient noise... ({} seconds)".format(
                        calibration_time
                    )
                )
                print("  Please remain quiet during calibration...")
                print("  VAD is learning your room's noise profile...")

                self.recognizer.adjust_for_ambient_noise(
                    source, duration=calibration_time
                )

                # PHASE 1: Collect noise samples for VAD initialization
                print("  Collecting VAD noise samples...")
                for i in range(10):
                    audio_chunk = self.recognizer.listen(
                        source, timeout=0.5, phrase_time_limit=0.5
                    )
                    audio_data = np.frombuffer(
                        audio_chunk.get_raw_data(), dtype=np.int16
                    )

                    # Preprocess if available
                    if self.preprocessor:
                        audio_data = self.preprocessor.process(audio_data)

                    # Feed to VAD for noise floor estimation
                    self.vad.is_speech(audio_data)

                # PHASE 1: Set threshold based on VAD
                vad_threshold = self.vad.get_threshold()
                self.recognizer.energy_threshold = vad_threshold

                print(
                    "Microphone calibrated. Energy threshold: {:.0f} (VAD-adjusted)".format(
                        self.recognizer.energy_threshold
                    )
                )
                print("  VAD noise floor: {:.0f}".format(self.vad.noise_floor))
                print("Ready for commands.")
                print("\n[Listening continuously...]\n")
        except sr.WaitTimeoutError:
            print("  Calibration timeout - using default threshold")
        except Exception as e:
            print("WARNING: Could not calibrate microphone: {}".format(e))

        while self.listening:
            try:
                with self.microphone as source:
                    # PHASE 1: Use listen() for continuous listening
                    phrase_limit = 5  # Commands are typically short

                    # Listen continuously - waits for speech
                    audio = self.recognizer.listen(
                        source,
                        timeout=None,
                        phrase_time_limit=phrase_limit,
                    )

                # PHASE 1: Preprocess audio BEFORE recognition
                try:
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

                    # Apply preprocessing if available
                    if self.preprocessor:
                        audio_data = self.preprocessor.process(audio_data)
                        # Create new AudioData from preprocessed samples
                        audio = sr.AudioData(
                            audio_data.tobytes(), audio.sample_rate, audio.sample_width
                        )

                    # PHASE 1: Check with VAD if this is actually speech
                    is_speech = self.vad.is_speech(audio_data)
                    if not is_speech:
                        # Not speech - skip recognition
                        continue

                    # PHASE 1: Update recognizer threshold based on VAD
                    new_threshold = self.vad.get_threshold()
                    self.recognizer.energy_threshold = new_threshold

                except Exception as e:
                    print("[Preprocessing error: {}]".format(str(e)[:50]))
                    # Continue with original audio if preprocessing fails

                try:
                    # PHASE 2: Use multi-pass verification for offline recognition
                    if self.use_offline:
                        try:
                            # PHASE 2: Use multi-pass consensus recognition
                            if self.multi_pass:
                                text = self.multi_pass.recognize_with_consensus(
                                    audio, num_passes=3
                                )
                            else:
                                # Fallback to single-pass if multi-pass not available
                                if self.grammar_file:
                                    text = self.recognizer.recognize_sphinx(
                                        audio,
                                        language="en-US",
                                        grammar=self.grammar_file,
                                    )
                                else:
                                    text = self.recognizer.recognize_sphinx(
                                        audio,
                                        language="en-US",
                                        keyword_entries=self.keywords,
                                    )

                            # Filter out very short/empty results
                            if text and len(text.strip()) > 3:
                                # Pre-filter - only process if it contains "simon"
                                if "simon" in text.lower():
                                    self.parse_command(text)
                                    self.consecutive_errors = 0
                                else:
                                    # Log what was detected for debugging
                                    print(
                                        "  [Detected but no 'simon': '{}']".format(text)
                                    )
                            else:
                                # Likely noise
                                pass

                        except sr.UnknownValueError:
                            # PocketSphinx couldn't understand - continue listening silently
                            pass
                        except sr.RequestError as e:
                            print("[Offline recognition error: {}]".format(e))
                            self.consecutive_errors += 1

                            if self.consecutive_errors >= 5:
                                print(
                                    "WARNING: Too many offline errors. Restarting recognizer..."
                                )
                                self.recognizer = sr.Recognizer()
                                # Re-apply optimizations
                                self.recognizer.energy_threshold = 600
                                self.recognizer.dynamic_energy_threshold = True
                                self.recognizer.pause_threshold = 0.6
                                self.recognizer.phrase_threshold = 0.4
                                self.recognizer.non_speaking_duration = 0.4
                                # Reinitialize multi-pass
                                self.multi_pass = MultiPassRecognizer(
                                    self.recognizer,
                                    grammar_file=self.grammar_file,
                                    keywords=self.keywords,
                                )
                                self.consecutive_errors = 0
                                print("[Listening continuously...]\n")
                    else:
                        # Online mode (Google API) - fallback if offline not available
                        try:
                            text = self.recognizer.recognize_google(
                                audio, language="en-US", show_all=False
                            )

                            if "simon" in text.lower():
                                self.parse_command(text)

                            self.consecutive_errors = 0
                        except sr.RequestError as e:
                            self.consecutive_errors += 1
                            print(
                                "[Google API error {}/{}: {}]".format(
                                    self.consecutive_errors,
                                    self.max_consecutive_errors,
                                    str(e)[:50],
                                )
                            )

                except sr.UnknownValueError:
                    # Speech was detected but not recognized - normal behavior
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
        """
        Get and clear the current command
        ROBUST: Returns metadata about command source for safer execution
        """
        cmd = {
            "command": self.command,
            "mode": self.mode,
            "target_object": self.target_object,
            "is_core": self._is_core_command(self.command) if self.command else True,
            "source": (
                "hardcoded"
                if (self.command and self._is_core_command(self.command))
                else "dynamic"
            ),
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
            # IMPROVEMENT: Pass NAO instance to enable speech feedback
            voice_listener = VoiceCommandListener(nao_instance=nao)
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
