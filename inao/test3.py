# -*- encoding: UTF-8 -*-
"""Example: Use Basic Awareness"""

import sys
import time
from naoqi import ALProxy


def main(robotIP):
    PORT = 9559

    try:
        basic_awareness = ALProxy("ALBasicAwareness", robotIP, PORT)
    except (Exception, e):
        print("Could not create proxy to ALBasicAwareness")
        print("Error was: ", e)
        sys.exit(1)

    # start
    basic_awareness.startAwareness()

    print("Basic Awareness started.")
    print("Use Ctrl+c to stop this script.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print
        print("Interrupted by user")
        print("Stopping...")

    # stop
    basic_awareness.stopAwareness()
    print("Basic Awareness stopped.")


if __name__ == "__main__":
    robotIp = "192.168.1.118"

    if len(sys.argv) <= 1:
        print(
            "Usage python albasicawareness_example.py robotIP (optional default: 127.0.0.1)"
        )
    else:
        robotIp = sys.argv[1]

    main(robotIp)
