# Quick Start Guide - V4 Voice Commands

## Installation Check
```bash
cd /home/georg/Desktop/hands_on_nao/inao

# Verify packages installed
~/.pyenv/versions/robocup_real/bin/pip list | grep -E "SpeechRecognition|pyaudio"

# Should show:
# pyaudio                0.2.14
# SpeechRecognition      3.9.0
```

## Test Voice Recognition (Without Robot)
```bash
~/.pyenv/versions/robocup_real/bin/python test_voice_recognition.py
```

## Run V4 (With Robot)
```bash
# Make sure robot IP is correct in script (default: 192.168.1.118)
~/.pyenv/versions/robocup_real/bin/python continuous_vision_test_v4.py
```

## Voice Commands

### Count Objects (Static)
**Say**: "how many bottles are there?"
**Robot**: Stops, counts, says "I see 2 bottles"

### Search for Objects (Rotating Head)
**Say**: "search for people"
**Robot**: Says "Searching for people", rotates head until found, says "Found it! I see 1 person"

### Stop Program
**Say**: "stop" or "quit"
**Robot**: Says "Stopping now" and exits

## Tips

### Speaking Clearly
- Speak at normal pace, not too fast or slow
- Stay within 1-2 feet of microphone
- Minimize background noise

### Object Names
- Use COCO class names (see coco.names file)
- Singular or plural both work ("bottle" or "bottles")
- Common: person, car, cup, chair, laptop, cell phone, bottle, dog, cat

### Troubleshooting
1. **No microphone detected**: Check `arecord -l`
2. **Can't understand speech**: Speak louder, reduce background noise
3. **Wrong object detected**: Check object is in COCO classes
4. **Robot not moving**: Check enable_head_rotation setting in code

## Configuration

Edit `continuous_vision_test_v4.py` at the bottom:

```python
# At line ~580
enable_head_rotation = False  # Start passive, controlled by voice
enable_voice_commands = True  # Enable voice control
target_objects = None         # Detect all objects (or specify list)
```

## How Voice Control Works

1. **Passive Mode** (default)
   - No head movement
   - Waiting for voice commands
   - Live video feed continues

2. **Query Mode** ("how many X")
   - Head stays still
   - Counts X in current frame
   - Reports count
   - Returns to passive

3. **Search Mode** ("search for X")
   - Enables head rotation
   - Scans left/right continuously
   - Stops when X is found
   - Reports count
   - Returns to passive

## Preserved V3 Features

All V3 functionality still works:
- ✅ YOLO object detection (80 classes)
- ✅ OCR number recognition
- ✅ Live video display
- ✅ Head rotation (now voice-controlled)
- ✅ Robot speech output
- ✅ Confidence thresholds

## Example Session

```
$ python continuous_vision_test_v4.py

=== Voice Command Listener Started ===
Say: 'how many [object]' to count objects
Say: 'search for [object]' to find objects
Say: 'stop' to quit
========================================

[Listening...]
Heard: 'how many bottles are there?'
Command parsed: MODE=query, OBJECT=bottle

Frame 42 - Acquisition delay: 0.123 seconds
  QUERY RESULT: I see 2 bottles
Robot: "I see 2 bottles"

[Listening...]
Heard: 'search for people'
Command parsed: MODE=search, OBJECT=person
Robot: "Searching for people"

Frame 45 - Head rotating left...
Frame 48 - Head rotating right...
Frame 51 - Detected: person (67.3%)
  SEARCH RESULT: Found it! I see 1 person
Robot: "Found it! I see 1 person"

[Listening...]
Heard: 'stop'
Command parsed: STOP
Robot: "Stopping now"

Unsubscribed from video feed. Processed 52 frames.
```

## Next Steps

1. Test voice recognition standalone
2. Connect to robot and test V4
3. Experiment with different objects
4. Adjust confidence thresholds if needed
5. Consider adding more voice commands

## Differences: V3 vs V4

| Aspect | V3 | V4 |
|--------|----|----|
| Control | Automatic | Voice-controlled |
| Head Rotation | Always on | On-demand (search mode) |
| Speech Output | All objects | Specific queries |
| Interaction | Passive | Interactive |
| Dependencies | None | +SpeechRecognition |
| Use Case | Monitoring | Q&A / Object search |
