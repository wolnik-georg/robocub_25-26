import sys

sys.path.insert(0, "/home/georg/Desktop/hands_on_nao/inao")
from inao import NAO

# Create NAO instance with robot's IP
nao = NAO("192.168.1.118")

# Test it
nao.tts.say("Hello, We are Team Electro technik")
