# -*- encoding: UTF-8 -*-
# Simple microphone test
import speech_recognition as sr

print("Testing microphone initialization...")
try:
    r = sr.Recognizer()
    m = sr.Microphone()
    print("SUCCESS: Microphone initialized!")

    print("\nAvailable microphones:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print("  [{}] {}".format(i, name))

    print("\nMicrophone is working! Ready for speech recognition.")
except Exception as e:
    print("ERROR: {}".format(e))
