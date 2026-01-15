# -*- encoding: UTF-8 -*-
# Test Voice Commands V4 - Standalone test without robot
# Tests speech recognition and command parsing for all V4 commands

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


def parse_command(text):
    """Parse voice command and extract intent - same logic as V4"""
    text = text.lower()
    print("\n" + "=" * 60)
    print("Heard: '{}'".format(text))
    print("=" * 60)

    # Check for "Simon says" prefix - required for all commands
    if not text.startswith("simon says"):
        print("  -> REJECTED: Missing 'Simon says' prefix")
        print("  -> All commands must start with 'Simon says'")
        return None

    # Remove "Simon says" prefix for further parsing
    text = text.replace("simon says", "").strip()
    print("  -> Processing command: '{}'".format(text))

    # Pattern: "how many [object]" or "how many [object]s are there"
    match = re.search(r"how many (\w+)", text)
    if match:
        obj = match.group(1)
        if obj.endswith("s"):
            obj = obj[:-1]
        print("  -> MATCHED: Vision Query")
        print("     Mode: Count objects")
        print("     Object: {}".format(obj))
        print("     Action: Robot would count {}s in current view".format(obj))
        return {"type": "query", "object": obj}

    # Pattern: "search for [object]" or "find [object]"
    match = re.search(r"(?:search for|find) (\w+)", text)
    if match:
        obj = match.group(1)
        if obj.endswith("s"):
            obj = obj[:-1]
        print("  -> MATCHED: Vision Search")
        print("     Mode: Active search with head rotation")
        print("     Object: {}".format(obj))
        print("     Action: Robot would rotate head to find {}".format(obj))
        return {"type": "search", "object": obj}

    # Physical movement commands
    if "go forward" in text or "move forward" in text or "walk forward" in text:
        print("  -> MATCHED: Movement - Forward")
        print("     Action: Robot would walk 0.3m forward")
        return {"type": "movement", "action": "move_forward"}
    if "go backward" in text or "move backward" in text or "go back" in text:
        print("  -> MATCHED: Movement - Backward")
        print("     Action: Robot would walk 0.2m backward")
        return {"type": "movement", "action": "move_backward"}
    if "go left" in text or "move left" in text or "step left" in text:
        print("  -> MATCHED: Movement - Left")
        print("     Action: Robot would step 0.2m left")
        return {"type": "movement", "action": "move_left"}
    if "go right" in text or "move right" in text or "step right" in text:
        print("  -> MATCHED: Movement - Right")
        print("     Action: Robot would step 0.2m right")
        return {"type": "movement", "action": "move_right"}
    if "turn left" in text or "rotate left" in text:
        print("  -> MATCHED: Movement - Turn Left")
        print("     Action: Robot would rotate 45 degrees left")
        return {"type": "movement", "action": "turn_left"}
    if "turn right" in text or "rotate right" in text:
        print("  -> MATCHED: Movement - Turn Right")
        print("     Action: Robot would rotate 45 degrees right")
        return {"type": "movement", "action": "turn_right"}

    # Posture commands
    if "stand up" in text or "stand" in text:
        print("  -> MATCHED: Posture - Stand")
        print("     Action: Robot would stand up")
        return {"type": "posture", "action": "stand"}
    if "sit down" in text or "sit" in text:
        print("  -> MATCHED: Posture - Sit")
        print("     Action: Robot would sit down")
        return {"type": "posture", "action": "sit"}
    if "crouch" in text or "crouch down" in text:
        print("  -> MATCHED: Posture - Crouch")
        print("     Action: Robot would crouch down")
        return {"type": "posture", "action": "crouch"}

    # Arm movements
    if "raise left arm" in text or "lift left arm" in text:
        print("  -> MATCHED: Arm Movement - Raise Left Arm")
        print("     Action: Robot would raise left arm to front")
        return {"type": "arm", "action": "raise_left_arm"}
    if "raise right arm" in text or "lift right arm" in text:
        print("  -> MATCHED: Arm Movement - Raise Right Arm")
        print("     Action: Robot would raise right arm to front")
        return {"type": "arm", "action": "raise_right_arm"}
    if "raise both arms" in text or "raise arms" in text:
        print("  -> MATCHED: Arm Movement - Raise Both Arms")
        print("     Action: Robot would raise both arms")
        return {"type": "arm", "action": "raise_both_arms"}

    # Head movements (explicit)
    if "look left" in text or "turn head left" in text:
        print("  -> MATCHED: Head Movement - Look Left")
        print("     Action: Robot would turn head 45 degrees left")
        return {"type": "head", "action": "head_left"}
    if "look right" in text or "turn head right" in text:
        print("  -> MATCHED: Head Movement - Look Right")
        print("     Action: Robot would turn head 45 degrees right")
        return {"type": "head", "action": "head_right"}
    if "look up" in text or "head up" in text:
        print("  -> MATCHED: Head Movement - Look Up")
        print("     Action: Robot would tilt head up")
        return {"type": "head", "action": "head_up"}
    if "look down" in text or "head down" in text:
        print("  -> MATCHED: Head Movement - Look Down")
        print("     Action: Robot would tilt head down")
        return {"type": "head", "action": "head_down"}

    # Wave gesture
    if "wave" in text or "say hello" in text:
        print("  -> MATCHED: Gesture - Wave")
        print("     Action: Robot would wave with right hand")
        return {"type": "gesture", "action": "wave"}

    # Pattern: "stop" or "quit"
    if "stop" in text or "quit" in text:
        print("  -> MATCHED: System Command - Stop")
        print("     Action: Robot would exit program")
        return {"type": "system", "action": "stop"}

    print("  -> NOT RECOGNIZED: Unknown command")
    print("     Try one of the supported commands (see help)")
    return None


def test_microphone():
    """Test if microphone is accessible"""
    print("\n" + "=" * 60)
    print("TESTING MICROPHONE")
    print("=" * 60)
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
    print("\n" + "=" * 60)
    print("TESTING SPEECH RECOGNITION")
    print("=" * 60)
    print("Adjusting for ambient noise... (wait 2 seconds)")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)

    print("Ambient noise adjustment complete.")
    print("\nSpeak a command! (listening for 10 seconds)")
    print("Try: 'Simon says how many bottles'")
    print("Or:  'Simon says go forward'")
    print("Or:  'Simon says wave'")

    try:
        with mic as source:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)

        print("\nProcessing audio...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        print("TIMEOUT: No speech detected")
        return None
    except sr.UnknownValueError:
        print("ERROR: Could not understand audio")
        return None
    except sr.RequestError as e:
        print("ERROR: Speech recognition service error: {}".format(e))
        return None
    except Exception as e:
        print("ERROR: Unexpected error: {}".format(e))
        return None


def print_help():
    """Print command examples"""
    print("\n" + "=" * 60)
    print("SUPPORTED VOICE COMMANDS (must start with 'Simon says')")
    print("=" * 60)
    print("\nVISION COMMANDS:")
    print("  - Simon says how many [bottles/people/cups/etc]")
    print("  - Simon says search for [bottles/people/cups/etc]")
    print("  - Simon says find [bottles/people/cups/etc]")
    print("\nMOVEMENT COMMANDS:")
    print("  - Simon says go forward")
    print("  - Simon says go backward / go back")
    print("  - Simon says go left / go right")
    print("  - Simon says turn left / turn right")
    print("\nPOSTURE COMMANDS:")
    print("  - Simon says stand / stand up")
    print("  - Simon says sit / sit down")
    print("  - Simon says crouch")
    print("\nARM COMMANDS:")
    print("  - Simon says raise left arm")
    print("  - Simon says raise right arm")
    print("  - Simon says raise both arms")
    print("\nHEAD COMMANDS:")
    print("  - Simon says look left / look right")
    print("  - Simon says look up / look down")
    print("\nGESTURES:")
    print("  - Simon says wave")
    print("  - Simon says say hello")
    print("\nSYSTEM:")
    print("  - Simon says stop / quit")
    print("\n" + "=" * 60)


def interactive_test():
    """Run interactive voice command testing"""
    print("\n" + "=" * 60)
    print("NAO V4 VOICE COMMAND TEST (NO ROBOT NEEDED)")
    print("=" * 60)

    print_help()

    # Test microphone
    recognizer, mic = test_microphone()
    if not recognizer or not mic:
        return

    print("\n" + "=" * 60)
    print("READY FOR VOICE TESTING!")
    print("=" * 60)
    print("\nYou can:")
    print("  1. Speak commands (will be recognized and parsed)")
    print("  2. Type 'h' for help")
    print("  3. Type 'q' to quit")
    print("  4. Press Enter to listen for next command")

    test_count = 0
    while True:
        print("\n" + "-" * 60)
        user_input = (
            raw_input("\nPress Enter to speak (or 'h' for help, 'q' to quit): ")
            .strip()
            .lower()
        )

        if user_input == "q":
            print("\nExiting test...")
            break
        elif user_input == "h":
            print_help()
            continue

        test_count += 1
        print("\n>>> TEST {} <<<".format(test_count))

        # Listen for speech
        text = test_speech_recognition(recognizer, mic)

        if text:
            # Parse the command
            result = parse_command(text)

            if result:
                print("\n  *** COMMAND SUCCESSFULLY RECOGNIZED ***")
                if result.get("type") == "system" and result.get("action") == "stop":
                    print("\n  (In real script, robot would exit now)")
            else:
                print("\n  *** COMMAND NOT RECOGNIZED ***")
        else:
            print("\n  *** NO SPEECH DETECTED ***")

    print("\n" + "=" * 60)
    print("TEST COMPLETE - Total tests: {}".format(test_count))
    print("=" * 60)


def manual_test():
    """Test command parsing with manual text input"""
    print("\n" + "=" * 60)
    print("MANUAL COMMAND PARSING TEST")
    print("=" * 60)
    print("\nType commands to test parsing (without saying them)")
    print("Type 'h' for help, 'q' to quit")

    test_commands = [
        "Simon says how many bottles",
        "Simon says search for people",
        "Simon says go forward",
        "Simon says wave",
        "how many bottles",  # Should fail - no "Simon says"
        "Simon says stop",
    ]

    print("\nSuggested test commands:")
    for i, cmd in enumerate(test_commands, 1):
        print("  {}. {}".format(i, cmd))

    while True:
        print("\n" + "-" * 60)
        text = raw_input("\nEnter command (or 'h' for help, 'q' to quit): ").strip()

        if text.lower() == "q":
            break
        elif text.lower() == "h":
            print_help()
            continue
        elif not text:
            continue

        parse_command(text)


def main():
    """Main test menu"""
    print("\n" + "=" * 60)
    print("NAO V4 VOICE COMMAND TESTING SUITE")
    print("=" * 60)
    print("\nSelect test mode:")
    print("  1. Interactive Voice Test (speak commands)")
    print("  2. Manual Text Test (type commands)")
    print("  3. Show Help & Examples")
    print("  4. Exit")

    while True:
        choice = raw_input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            if not SPEECH_AVAILABLE:
                print("\nERROR: Speech recognition not available!")
                continue
            interactive_test()
            break
        elif choice == "2":
            manual_test()
            break
        elif choice == "3":
            print_help()
        elif choice == "4":
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
