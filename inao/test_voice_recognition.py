# -*- encoding: UTF-8 -*-
# Test script to verify speech recognition setup
# Run this to test microphone and speech recognition without connecting to robot

import sys
import time
import re

try:
    import speech_recognition as sr

    SPEECH_AVAILABLE = True
    print("SUCCESS: speech_recognition module imported")
except ImportError:
    SPEECH_AVAILABLE = False
    print("ERROR: speech_recognition not available")
    print("Install with: pip install SpeechRecognition pyaudio")
    sys.exit(1)


def test_microphone():
    """Test if microphone is accessible"""
    print("\n=== Testing Microphone ===")
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print("SUCCESS: Microphone initialized")

        # List available microphones
        print("\nAvailable microphones:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print("  [{}] {}".format(index, name))

        return recognizer, mic
    except Exception as e:
        print("ERROR: Failed to initialize microphone")
        print("Error: {}".format(e))
        return None, None


def test_speech_recognition(recognizer, mic):
    """Test speech recognition"""
    print("\n=== Testing Speech Recognition ===")
    print("Adjusting for ambient noise... (wait 2 seconds)")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)

    print("Ambient noise adjustment complete.")
    print("\nSay something! (listening for 5 seconds)")

    try:
        with mic as source:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

        print("Processing audio...")
        text = recognizer.recognize_google(audio)
        print("You said: '{}'".format(text))
        return text
    except sr.WaitTimeoutError:
        print("No speech detected (timeout)")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Speech recognition error: {}".format(e))
        return None
    except Exception as e:
        print("Unexpected error: {}".format(e))
        return None


def test_command_parsing(text):
    """Test command parsing logic"""
    print("\n=== Testing Command Parsing ===")

    if not text:
        print("No text to parse")
        return

    text = text.lower()
    print("Input text: '{}'".format(text))

    # Check for "Simon says" prefix
    if not text.startswith("simon says"):
        print("NO MATCH: Missing 'Simon says' prefix")
        print("All commands must start with 'Simon says'")
        return

    # Remove "Simon says" prefix
    text = text.replace("simon says", "").strip()
    print("Processing: '{}'".format(text))

    # Test "how many" pattern
    match = re.search(r"how many (\w+)", text)
    if match:
        obj = match.group(1)
        if obj.endswith("s"):
            obj = obj[:-1]
        print("MATCHED: Query mode")
        print("  Object: {}".format(obj))
        return

    # Test "search for" pattern
    match = re.search(r"(?:search for|find) (\w+)", text)
    if match:
        obj = match.group(1)
        if obj.endswith("s"):
            obj = obj[:-1]
        print("MATCHED: Search mode")
        print("  Object: {}".format(obj))
        return

    # Test "stop" pattern
    if "stop" in text or "quit" in text:
        print("MATCHED: Stop command")
        return

    print("NO MATCH: Command not recognized")
    print("Try saying:")
    print("  - 'Simon says how many bottles'")
    print("  - 'Simon says search for people'")
    print("  - 'Simon says stop'")


def main():
    print("=" * 50)
    print("NAO Voice Command Test")
    print("=" * 50)

    # Test microphone
    recognizer, mic = test_microphone()
    if not recognizer or not mic:
        return

    print("\n" + "=" * 50)
    print("Ready for speech recognition test!")
    print("=" * 50)

    # Run multiple tests
    for i in range(3):
        print("\n--- Test {} of 3 ---".format(i + 1))
        text = test_speech_recognition(recognizer, mic)
        if text:
            test_command_parsing(text)

        if i < 2:
            print("\nWaiting 2 seconds before next test...")
            time.sleep(2)

    print("\n" + "=" * 50)
    print("Voice recognition test complete!")
    print("=" * 50)
    print("\nIf all tests passed, you can run continuous_vision_test_v4.py")


if __name__ == "__main__":
    main()
