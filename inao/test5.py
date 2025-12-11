# -*- encoding: UTF-8 -*-
"""Example: Use startTracker Method to Track a Face"""

import sys
import time
from naoqi import ALProxy


def main(robotIP):
    PORT = 9559

    try:
        faceTracker = ALProxy("ALFaceTracker", robotIP, PORT)
    except Exception:
        print("Could not create proxy to ALFaceTracker")
        print("Error was: ")
        sys.exit(1)

    # Start the tracker by specifying the target to track
    targetName = "Face"
    faceWidth = 0.1
    faceTracker.startTracker(targetName, faceWidth)

    print("ALFaceTracker successfully started.")
    print("Use Ctrl+c to stop this script.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print
        print("Interrupted by user")
        print("Stopping...")

    # Stop the tracker
    faceTracker.stopTracker()
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
