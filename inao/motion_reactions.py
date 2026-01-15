import sys
import os
import math
import time
import json
import subprocess

from naoqi import ALProxy

# file_path = os.path.dirname(os.path.abspath(__file__))
# georg_path = os.path.join(file_path, "..", "georgs-scripte")
# sys.path.append(georg_path)
# from advanced_speech_recognition import EnhancedSpeechRecognitionDemo


class MotionReactions(object):
    def __init__(self, robotIP="192.168.1.118", PORT=9559):
        self.robotIP = robotIP
        self.PORT = PORT

    def wakeUp(self):
        """
        Wake up the robot (stand up).
        """
        motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)
        motionProxy.wakeUp()

    def rest(self):
        """
        Rest the robot (sit down).
        """
        motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)
        motionProxy.rest()

    def posture(self, posture_name="stand", speed=1.0):
        """
        Set the robot in a given posture
        :param posture_name: name of the posture
        :param speed: speed of the movement
        :return: None

        Possible postures:
        "Stand"
        "StandInit"
        "Crouch"
        "Sit"

        """
        ttsProxy = ALProxy("ALTextToSpeech", self.robotIP, self.PORT)
        postureProxy = ALProxy("ALRobotPosture", self.robotIP, self.PORT)
        motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)
        postureProxy.goToPosture(posture_name, speed)
        ttsProxy.say("Movement {} complete!".format(posture_name))
        motionProxy.rest()

    def move_position(self, x=0.0, y=0.0, theta=0.0):
        """
        Move the robot
        :param x: forward/backward movement in meters
        :param y: left/right movement in meters
        :param theta: rotation in radians
        :return: None
        """
        ttsProxy = ALProxy("ALTextToSpeech", self.robotIP, self.PORT)
        motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)
        motionProxy.wakeUp()
        motionProxy.moveTo(x, y, theta)
        ttsProxy.say("Movement complete!")
        motionProxy.rest()

    def move_joint(self, joint_name="HeadYaw", angle=0.0, speed=0.1, waitingtime=2.0):
        """
        Move a specific joint
        :param joint_name: name of the joint
        :param angle: angle in radians
        :param speed: speed of the movement (0.0 to 1.0)
        :return: None
        """
        ttsProxy = ALProxy("ALTextToSpeech", self.robotIP, self.PORT)
        motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)
        # Removed motionProxy.wakeUp() and motionProxy.rest() to avoid repeated wake/rest in loops
        motionProxy.setStiffnesses(joint_name, 1.0)
        motionProxy.setAngles(joint_name, angle, speed)
        ttsProxy.say("Movement of joint {} complete!".format(joint_name))
        time.sleep(waitingtime)
        motionProxy.setStiffnesses(joint_name, 0.0)
        # Removed motionProxy.rest()


def main():
    print("Starting Simple Reaction")

    # Default IP
    robotIp = "192.168.1.118"

    # Check if IP is provided as argument
    if len(sys.argv) > 1:
        robotIp = sys.argv[1]

    # # Subprocess aufrufen
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # python3 = "/usr/bin/python3"
    # script_py3 = os.path.join(BASE_DIR, "extract_command.py")
    # audio_file = os.path.join(BASE_DIR, "recorded_command.wav")

    # # Sicherheitschecks (einmal drin lassen!)
    # assert os.path.isfile(script_py3), script_py3
    # assert os.path.isfile(audio_file), audio_file

    # output = subprocess.check_output([python3, script_py3, audio_file])
    # result = json.loads(output)
    # print("Extracted Command:", result)

    example_dict = {
        # Predifined postures
        "StandZero": {
            "function": "posture",
            "params": {"posture_name": "StandZero", "speed": 1.0},
        },
        "StandInit": {
            "function": "posture",
            "params": {"posture_name": "StandInit", "speed": 1.0},
        },
        "Stand": {
            "function": "posture",
            "params": {"posture_name": "Stand", "speed": 1.0},
        },
        "Crouch": {
            "function": "posture",
            "params": {"posture_name": "Crouch", "speed": 1.0},
        },
        "Sit": {"function": "posture", "params": {"posture_name": "Sit", "speed": 1.0}},
        "SitRelax": {
            "function": "posture",
            "params": {"posture_name": "SitRelax", "speed": 1.0},
        },
        "LyingBelly": {
            "function": "posture",
            "params": {"posture_name": "LyingBelly", "speed": 1.0},
        },
        "LyingBack": {
            "function": "posture",
            "params": {"posture_name": "LyingBack", "speed": 1.0},
        },
        # Custom movements
        ## Move Feet
        "MoveForward": {
            "function": "move_position",
            "params": {"x": 0.5, "y": 0.0, "theta": 0.0},
        },
        "MoveBackward": {
            "function": "move_position",
            "params": {"x": -0.2, "y": 0.0, "theta": 0.0},
        },
        "MoveRight": {
            "function": "move_position",
            "params": {"x": 0.0, "y": -0.2, "theta": 0.0},
        },
        "MoveLeft": {
            "function": "move_position",
            "params": {"x": 0.0, "y": 0.2, "theta": 0.0},
        },
        "TurnRight": {
            "function": "move_position",
            "params": {"x": 0.0, "y": 0.0, "theta": -math.pi},
        },
        "TurnLeft": {
            "function": "move_position",
            "params": {"x": 0.0, "y": 0.0, "theta": math.pi / 4},
        },
        ## Move Head
        "RotateHeadRight": {
            "function": "move_joint",
            "params": {
                "joint_name": "HeadYaw",
                "angle": -math.pi / 4,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "RotateHeadLeft": {
            "function": "move_joint",
            "params": {
                "joint_name": "HeadYaw",
                "angle": math.pi / 4,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "MoveHeadRight": {
            "function": "move_joint",
            "params": {
                "joint_name": "HeadPitch",
                "angle": 10,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        ## Move Arms
        "LiftLeftArmFront": {
            "function": "move_joint",
            "params": {
                "joint_name": "LShoulderPitch",
                "angle": 0,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "LiftRightArmFront": {
            "function": "move_joint",
            "params": {
                "joint_name": "RShoulderPitch",
                "angle": 0,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "LiftLeftArmSide": {
            "function": "move_joint",
            "params": {
                "joint_name": "LShoulderRoll",
                "angle": 60,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "LiftRightArmSide": {
            "function": "move_joint",
            "params": {
                "joint_name": "RShoulderRoll",
                "angle": -60,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "StretchLeftElbow": {
            "function": "move_joint",
            "params": {
                "joint_name": "LElbowRoll",
                "angle": -2,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "BendLeftElbow": {
            "function": "move_joint",
            "params": {
                "joint_name": "LElbowRoll",
                "angle": -88,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "StretchRightElbow": {
            "function": "move_joint",
            "params": {
                "joint_name": "RElbowRoll",
                "angle": 2,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "BendRightElbow": {
            "function": "move_joint",
            "params": {
                "joint_name": "RElbowRoll",
                "angle": 88,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "TwistLeftWrist": {
            "function": "move_joint",
            "params": {
                "joint_name": "LWristYaw",
                "angle": -90,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        "TwistRightWrist": {
            "function": "move_joint",
            "params": {
                "joint_name": "RWristYaw",
                "angle": 90,
                "speed": 0.1,
                "waitingtime": 2.0,
            },
        },
        ## Add more custom movements as needed
    }

    motion = MotionReactions(robotIp, 9559)
    command = "RotateHeadLeft"  # Change this to test different reactions
    if command:
        details = example_dict[command]
        func = getattr(motion, details["function"])
        params = details["params"]
        func(**params)

    elif result:
        pass

    else:
        # No command provided, you can implement a default behavior or interaction here
        pass


if __name__ == "__main__":
    print("Running Reaction Motion Script")
    main()
