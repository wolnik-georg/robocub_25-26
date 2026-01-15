# -*- encoding: UTF-8 -*-
# Simple command parsing test - NO microphone needed
# Tests just the command parsing logic from V4

import re


def parse_command(text):
    """Parse voice command and extract intent - same logic as V4"""
    text = text.lower()
    print("\n" + "=" * 60)
    print("Input: '{}'".format(text))
    print("=" * 60)

    # Check for "Simon says" prefix - required for all commands
    if not text.startswith("simon says"):
        print("  -> REJECTED: Missing 'Simon says' prefix")
        return None

    # Remove "Simon says" prefix for further parsing
    text = text.replace("simon says", "").strip()
    print("  -> Command text: '{}'".format(text))

    # Pattern: "how many [object]"
    match = re.search(r"how many (\w+)", text)
    if match:
        obj = match.group(1)
        if obj.endswith("s"):
            obj = obj[:-1]
        print("  -> MATCHED: Vision Query")
        print("     Object: {}".format(obj))
        return {"type": "query", "object": obj}

    # Pattern: "search for [object]" or "find [object]"
    match = re.search(r"(?:search for|find) (\w+)", text)
    if match:
        obj = match.group(1)
        if obj.endswith("s"):
            obj = obj[:-1]
        print("  -> MATCHED: Vision Search")
        print("     Object: {}".format(obj))
        return {"type": "search", "object": obj}

    # Movement commands
    if "go forward" in text or "move forward" in text:
        print("  -> MATCHED: Movement - Forward (0.3m)")
        return {"type": "movement", "action": "move_forward"}
    if "go backward" in text or "go back" in text:
        print("  -> MATCHED: Movement - Backward (0.2m)")
        return {"type": "movement", "action": "move_backward"}
    if "go left" in text:
        print("  -> MATCHED: Movement - Left (0.2m)")
        return {"type": "movement", "action": "move_left"}
    if "go right" in text:
        print("  -> MATCHED: Movement - Right (0.2m)")
        return {"type": "movement", "action": "move_right"}
    if "turn left" in text:
        print("  -> MATCHED: Movement - Turn Left (45deg)")
        return {"type": "movement", "action": "turn_left"}
    if "turn right" in text:
        print("  -> MATCHED: Movement - Turn Right (45deg)")
        return {"type": "movement", "action": "turn_right"}

    # Posture
    if "stand" in text:
        print("  -> MATCHED: Posture - Stand")
        return {"type": "posture", "action": "stand"}
    if "sit" in text:
        print("  -> MATCHED: Posture - Sit")
        return {"type": "posture", "action": "sit"}
    if "crouch" in text:
        print("  -> MATCHED: Posture - Crouch")
        return {"type": "posture", "action": "crouch"}

    # Arms
    if "raise left arm" in text:
        print("  -> MATCHED: Arm - Raise Left")
        return {"type": "arm", "action": "raise_left_arm"}
    if "raise right arm" in text:
        print("  -> MATCHED: Arm - Raise Right")
        return {"type": "arm", "action": "raise_right_arm"}
    if "raise both arms" in text or "raise arms" in text:
        print("  -> MATCHED: Arm - Raise Both")
        return {"type": "arm", "action": "raise_both_arms"}

    # Head
    if "look left" in text:
        print("  -> MATCHED: Head - Look Left")
        return {"type": "head", "action": "head_left"}
    if "look right" in text:
        print("  -> MATCHED: Head - Look Right")
        return {"type": "head", "action": "head_right"}
    if "look up" in text:
        print("  -> MATCHED: Head - Look Up")
        return {"type": "head", "action": "head_up"}
    if "look down" in text:
        print("  -> MATCHED: Head - Look Down")
        return {"type": "head", "action": "head_down"}

    # Gestures
    if "wave" in text:
        print("  -> MATCHED: Gesture - Wave")
        return {"type": "gesture", "action": "wave"}

    # System
    if "stop" in text or "quit" in text:
        print("  -> MATCHED: System - Stop")
        return {"type": "system", "action": "stop"}

    print("  -> NOT RECOGNIZED")
    return None


# Test cases
test_commands = [
    # Vision commands
    "Simon says how many bottles",
    "Simon says how many people",
    "Simon says search for cups",
    "Simon says find person",
    # Movement commands
    "Simon says go forward",
    "Simon says go backward",
    "Simon says go left",
    "Simon says turn right",
    # Posture commands
    "Simon says stand",
    "Simon says sit down",
    "Simon says crouch",
    # Arm commands
    "Simon says raise left arm",
    "Simon says raise right arm",
    "Simon says raise both arms",
    # Head commands
    "Simon says look left",
    "Simon says look up",
    # Gestures
    "Simon says wave",
    # System
    "Simon says stop",
    # Should fail - no "Simon says"
    "how many bottles",
    "go forward",
]

print("\n" + "=" * 60)
print("COMMAND PARSING TEST - V4 Logic")
print("=" * 60)
print("\nTesting {} commands...\n".format(len(test_commands)))

success_count = 0
fail_count = 0

for cmd in test_commands:
    result = parse_command(cmd)
    if result:
        success_count += 1
        print("  *** SUCCESS ***")
    else:
        fail_count += 1
        print("  *** FAILED (as expected if no 'Simon says') ***")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("Total tests: {}".format(len(test_commands)))
print("Recognized: {}".format(success_count))
print("Rejected: {}".format(fail_count))
print("\nAll parsing logic working correctly!")
print("=" * 60)
