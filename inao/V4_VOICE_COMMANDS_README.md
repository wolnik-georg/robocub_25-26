# Continuous Vision Test V4 - Voice Commands

## Overview
V4 extends V3 with **speech recognition** for interactive voice-controlled object detection and search.

## New Features

### 1. Voice Command System
- **Microphone**: Uses your laptop's microphone
- **Recognition**: Google Speech Recognition API (requires internet)
- **Threading**: Runs in background, doesn't block vision processing

### 2. Voice Commands

#### Query Mode: "how many [object]"
- **Example**: "how many bottles are there?"
- **Behavior**: 
  - Stops head rotation
  - Counts specified object in current camera view
  - Robot speaks result: "I see 2 bottles" or "I see no bottles"
  - Returns to passive mode

#### Search Mode: "search for [object]"
- **Example**: "search for people" or "find bottles"
- **Behavior**:
  - Enables head rotation
  - Robot says "Searching for [object]"
  - Continuously rotates head and scans
  - When object found, robot says "Found it! I see X [object]s"
  - Stops rotation and returns to passive mode

#### Stop Command: "stop" or "quit"
- Exits the program gracefully

### 3. Operating Modes

#### Passive Mode (default with voice commands enabled)
- No automatic head rotation
- Continuous vision processing
- Displays video feed
- Waits for voice commands

#### Query Mode (triggered by "how many" command)
- Static position, no head movement
- Single frame analysis
- Reports count immediately

#### Search Mode (triggered by "search for" command)
- Active head rotation (left/right scanning)
- Continuous searching until target found
- Auto-stops when object detected

## Installation

### Dependencies
```bash
# Already installed in robocup_real environment:
pip install SpeechRecognition pyaudio
```

### System Requirements
- Working microphone
- Internet connection (for Google Speech Recognition)
- PortAudio library (usually pre-installed on Linux)

## Usage

### Running V4
```bash
cd /home/georg/Desktop/hands_on_nao/inao
~/.pyenv/versions/robocup_real/bin/python continuous_vision_test_v4.py
```

### Configuration
In the script's `__main__` section:
```python
enable_head_rotation = False  # Start in passive mode
enable_voice_commands = True  # Enable voice control
target_objects = None         # Detect all 80 COCO objects
```

### Example Session
```
[Listening...]
User: "how many bottles are there?"
Robot: "I see 3 bottles"

[Listening...]
User: "search for people"
Robot: "Searching for people"
[Head rotates left/right]
Robot: "Found it! I see 2 people"

[Listening...]
User: "stop"
Robot: "Stopping now"
```

## How It Works

### VoiceCommandListener Class
- Runs in separate thread
- Continuously listens for speech
- Parses commands using regex patterns
- Sets command flags for main loop

### Command Parsing
```python
# Patterns recognized:
"how many (\w+)"           # Query mode
"search for (\w+)"         # Search mode  
"find (\w+)"               # Search mode (alternative)
"stop" / "quit"            # Exit
```

### State Machine
1. **Listening** → Waiting for voice input
2. **Command Detected** → Parse and set mode
3. **Execute** → Query (count) or Search (rotate + find)
4. **Report** → Robot speaks result
5. **Return to Passive** → Wait for next command

## Technical Details

### Speech Recognition
- **Library**: `speech_recognition` (Python 2.7 compatible)
- **Engine**: Google Web Speech API
- **Audio**: PyAudio for microphone access
- **Timeout**: 5 seconds listening window
- **Ambient Noise**: Auto-adjusts for background noise

### Object Detection
- Uses same YOLO + OCR as V3
- Filters for specific objects in query/search modes
- Counts instances of target object

### Head Movement
- **Search Mode**: Rotates ±45° (pi/4 radians)
- **Timing**: Every 3 frames (2-3 images per position)
- **Speed**: 0.1 rad/s with 1.5s wait time
- **Stabilization**: 1.0s delay after movement

### Thread Safety
- Voice listener runs in daemon thread
- Command flags are simple primitives (no locks needed)
- Main loop checks flags each iteration

## Differences from V3

| Feature | V3 | V4 |
|---------|----|----|
| **Head Rotation** | Always on (if enabled) | Controlled by voice commands |
| **Speech Output** | Continuous (all objects) | Context-aware (query/search results) |
| **Interaction** | Passive observation | Active voice control |
| **Default Mode** | Head rotation enabled | Passive (waiting for commands) |
| **Dependencies** | None extra | +SpeechRecognition, +pyaudio |

## Troubleshooting

### "speech_recognition not available"
```bash
pip install SpeechRecognition pyaudio
```

### Microphone not working
```bash
# Test microphone
arecord -l

# Check permissions
sudo usermod -a -G audio $USER
```

### "Could not understand audio"
- Speak clearly and close to microphone
- Reduce background noise
- Check internet connection

### Object names not recognized
- Use singular form: "bottle" not "bottles"
- Use COCO class names: person, car, cup, chair, etc.
- See coco.names file for full list

## COCO Object Classes (80 total)
Common objects you can query/search:
- **People**: person
- **Vehicles**: car, truck, bus, bicycle, motorcycle
- **Animals**: dog, cat, bird, horse, cow
- **Food**: apple, banana, orange, pizza, donut
- **Furniture**: chair, couch, bed, table
- **Electronics**: laptop, cell phone, tv, keyboard, mouse
- **Kitchen**: bottle, cup, fork, knife, spoon, bowl
- **Sports**: ball, baseball bat, tennis racket, skateboard

Full list in: `./models/coco.names`

## Future Enhancements
- [ ] Add "show me [object]" command (point to object location)
- [ ] Voice feedback during search ("still looking...")
- [ ] Custom wake word ("Hey NAO")
- [ ] Offline speech recognition
- [ ] Multi-language support
- [ ] Save detected objects to file
