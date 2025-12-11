# -*- encoding: UTF-8 -*-
"""Enhanced Example: Use startTracker Method to Track a Face with Interaction"""

import sys
import time
from naoqi import ALProxy


def main(robotIP):
    PORT = 9559

    try:
        faceTracker = ALProxy("ALFaceTracker", robotIP, PORT)
        ttsProxy = ALProxy("ALTextToSpeech", robotIP, PORT)  # Added for interaction
        memoryProxy = ALProxy("ALMemory", robotIP, PORT)  # Added to detect faces
    except Exception:
        print("Could not create proxy to ALFaceTracker, ALTextToSpeech, or ALMemory")
        print("Error was: ")
        sys.exit(1)

    # Enhanced: Welcome message
    ttsProxy.say("I'm looking for faces! Stand in front of me.")

    # Start the tracker by specifying the target to track
    targetName = "Face"
    faceWidth = 0.1
    faceTracker.startTracker(targetName, faceWidth)

    print("ALFaceTracker successfully started.")
    print("Use Ctrl+c to stop this script.")

    faceDetected = False
    try:
        while True:
            # Enhanced: Check if a face is currently detected
            facePosition = memoryProxy.getData("FaceDetected")
            if facePosition and len(facePosition) > 0 and not faceDetected:
                ttsProxy.say("I see you! Hello there!")
                faceDetected = True
                print("Face detected - greeting given")
            elif not facePosition or len(facePosition) == 0:
                faceDetected = False

            time.sleep(1)
    except KeyboardInterrupt:
        print
        print("Interrupted by user")
        print("Stopping...")

    # Stop the tracker
    faceTracker.stopTracker()
    ttsProxy.say("Face tracking stopped. Goodbye!")
    print("ALFaceTracker stopped.")


if __name__ == "__main__":
    robotIp = "192.168.1.118"

    if len(sys.argv) <= 1:
        print(
            "Usage python alfacetracker_start.py robotIP (optional default: 127.0.0.1)"
        )
    else:
        robotIp = sys.argv[1]

    main(robotIp)
