# V4 Complete Voice Command Reference

## All Available Commands (Must start with "Simon says")

### 🔍 **Vision & Detection Commands**

#### Count Objects (Query Mode)
- `"Simon says how many bottles"` → Count bottles in current view
- `"Simon says how many people"` → Count people in current view
- `"Simon says how many cups"` → Count cups in current view
- Works with any COCO object (80 classes)

#### Search for Objects (Active Search Mode)
- `"Simon says search for bottles"` → Rotate head to find bottles
- `"Simon says find people"` → Rotate head to find people
- `"Simon says search for chairs"` → Rotate head to find chairs
- Robot rotates head until object found, then reports count

---

### 🚶 **Movement Commands**

#### Walk/Move
- `"Simon says go forward"` → Move 0.3m forward
- `"Simon says move forward"` → Same as above
- `"Simon says walk forward"` → Same as above
- `"Simon says go backward"` → Move 0.2m backward
- `"Simon says move backward"` → Same as above
- `"Simon says go back"` → Same as above
- `"Simon says go left"` → Step 0.2m left
- `"Simon says move left"` → Same as above
- `"Simon says step left"` → Same as above
- `"Simon says go right"` → Step 0.2m right
- `"Simon says move right"` → Same as above
- `"Simon says step right"` → Same as above

#### Turn/Rotate
- `"Simon says turn left"` → Rotate 45° left
- `"Simon says rotate left"` → Same as above
- `"Simon says turn right"` → Rotate 45° right
- `"Simon says rotate right"` → Same as above

---

### 🧍 **Posture Commands**

- `"Simon says stand"` → Stand up to standing position
- `"Simon says stand up"` → Same as above
- `"Simon says sit"` → Sit down
- `"Simon says sit down"` → Same as above
- `"Simon says crouch"` → Crouch down
- `"Simon says crouch down"` → Same as above

---

### 💪 **Arm Commands**

#### Raise Arms
- `"Simon says raise left arm"` → Raise left arm to front
- `"Simon says lift left arm"` → Same as above
- `"Simon says raise right arm"` → Raise right arm to front
- `"Simon says lift right arm"` → Same as above
- `"Simon says raise both arms"` → Raise both arms
- `"Simon says raise arms"` → Same as above

#### Wave Gesture
- `"Simon says wave"` → Wave with right hand (raises arm, waves wrist 3x, lowers)
- `"Simon says say hello"` → Same as above

---

### 👀 **Head/Look Commands**

- `"Simon says look left"` → Turn head 45° left
- `"Simon says turn head left"` → Same as above
- `"Simon says look right"` → Turn head 45° right
- `"Simon says turn head right"` → Same as above
- `"Simon says look up"` → Tilt head up
- `"Simon says head up"` → Same as above
- `"Simon says look down"` → Tilt head down
- `"Simon says head down"` → Same as above

---

### 🛑 **System Commands**

- `"Simon says stop"` → Exit program
- `"Simon says quit"` → Same as above
- Press **`q`** in video window → Also exits
- Press **`Ctrl+C`** → Also exits

---

## Command Categories Summary

| Category | Commands | Robot Action |
|----------|----------|--------------|
| **Vision** | how many, search for | Object detection & counting |
| **Movement** | go, move, walk, step | Physical locomotion |
| **Turning** | turn, rotate | Rotate in place |
| **Posture** | stand, sit, crouch | Body position changes |
| **Arms** | raise, lift | Arm movements |
| **Gestures** | wave, say hello | Pre-programmed gestures |
| **Head** | look, turn head | Head orientation |
| **System** | stop, quit | Program control |

---

## Usage Examples

### Example Session 1: Object Search
```
[Listening...]
You: "Simon says search for bottles"
Robot: "Searching for bottles"
[Head rotates left/right scanning]
Robot: "Found it! I see 2 bottles"
```

### Example Session 2: Movement Sequence
```
[Listening...]
You: "Simon says go forward"
Robot: "Moving forward"
[Walks 0.3m forward]

[Listening...]
You: "Simon says turn left"
Robot: "Turning left"
[Rotates 45° left]

[Listening...]
You: "Simon says wave"
Robot: "Hello! Waving!"
[Raises arm and waves]
```

### Example Session 3: Inspection Task
```
[Listening...]
You: "Simon says look left"
Robot: "Looking left"
[Turns head left]

[Listening...]
You: "Simon says how many people"
Robot: "I see 3 people"

[Listening...]
You: "Simon says look right"
Robot: "Looking right"
[Turns head right]

[Listening...]
You: "Simon says how many cups"
Robot: "I see no cups"
```

---

## Command Patterns

### Alternative Phrasings (All Work)
- Forward: "go forward" | "move forward" | "walk forward"
- Backward: "go backward" | "move backward" | "go back"
- Left/Right: "go left" | "move left" | "step left"
- Turn: "turn left" | "rotate left"
- Stand: "stand" | "stand up"
- Sit: "sit" | "sit down"
- Arms: "raise left arm" | "lift left arm"
- Head: "look left" | "turn head left"
- Wave: "wave" | "say hello"

### ⚠️ Important Notes
1. **"Simon says" is REQUIRED** for all commands
2. Without "Simon says" → Command ignored
3. Commands are **case-insensitive** ("SIMON SAYS" works)
4. Robot speaks before and after movements
5. Vision continues running during movements
6. Multiple commands can be chained (say one, wait, say next)

---

## Technical Details

### Movement Parameters
- Forward: 0.3 meters
- Backward: 0.2 meters
- Sideways: 0.2 meters
- Turn angle: 45° (π/4 radians)
- Head rotation: 45° (π/4 radians)
- Head pitch: ±0.3 radians

### Speech Feedback
- Robot confirms every command verbally
- Example: "Moving forward", "I see 2 bottles", "Raising left arm"
- Helps user know command was recognized

### Safety
- Robot wakes up before movements
- Robot rests after movements (sit position)
- Posture commands manage stiffness automatically
- Head movements are gentle and safe

---

## COCO Object Classes (for Vision Commands)

**Common objects you can count/search:**
- **People**: person
- **Furniture**: chair, couch, bed, table, desk
- **Electronics**: laptop, cell phone, tv, keyboard, mouse, remote
- **Kitchen**: bottle, cup, fork, knife, spoon, bowl, wine glass
- **Food**: apple, banana, orange, pizza, donut, cake
- **Vehicles**: car, truck, bus, bicycle, motorcycle
- **Animals**: dog, cat, bird, horse, cow, sheep
- **Sports**: ball, baseball bat, tennis racket, skateboard

**Full list**: See `./models/coco.names` (80 total classes)

---

## Running the Script

```bash
cd /home/georg/Desktop/hands_on_nao/inao
~/.pyenv/versions/robocup_real/bin/python continuous_vision_test_v4.py
```

**Requirements:**
- NAO robot powered on and connected
- Microphone working on laptop
- Internet connection (for Google Speech Recognition)
- Models downloaded (yolov3-tiny.cfg, yolov3-tiny.weights, coco.names)

---

## Quick Reference Card

| Say This | Robot Does |
|----------|------------|
| Simon says how many X | Counts X objects |
| Simon says search for X | Finds X objects |
| Simon says go forward | Walks forward |
| Simon says turn left | Rotates left |
| Simon says stand | Stands up |
| Simon says wave | Waves hand |
| Simon says look left | Turns head left |
| Simon says stop | Exits program |

**Remember: ALL commands need "Simon says" at the start!** 🎮
