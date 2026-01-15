# Simon Says - Voice Command Prefix

## Overview
All voice commands in V4 now require the **"Simon says"** prefix to prevent accidental triggering.

## Why "Simon Says"?
- **Prevents False Positives**: Background conversation won't trigger commands
- **Intentional Activation**: You must explicitly say "Simon says" to control the robot
- **Clear Interaction**: Robot only responds to deliberate commands
- **Fun Factor**: Classic game reference makes it more engaging

## Updated Voice Commands

### ✅ Valid Commands (WITH "Simon says")

#### Count Objects
```
"Simon says how many bottles are there?"
"Simon says how many people?"
"Simon says how many chairs are there?"
```

#### Search for Objects
```
"Simon says search for bottles"
"Simon says find people"
"Simon says search for cups"
```

#### Stop Program
```
"Simon says stop"
"Simon says quit"
```

### ❌ Invalid Commands (WITHOUT "Simon says")

These will be **ignored**:
```
"how many bottles?"          -> Ignored
"search for people"          -> Ignored
"stop"                       -> Ignored
```

You'll see in the console:
```
Heard: 'how many bottles'
  -> Ignored (missing 'Simon says' prefix)
```

## How It Works

### Command Processing Flow
1. **Speech detected** → Convert audio to text
2. **Check prefix** → Must start with "Simon says"
3. **If no prefix** → Ignore and continue listening
4. **If prefix found** → Remove prefix and parse command
5. **Execute** → Run the requested action

### Example Session
```
[Listening...]
Heard: 'how many bottles'
  -> Ignored (missing 'Simon says' prefix)

[Listening...]
Heard: 'Simon says how many bottles are there?'
  -> Processing command: 'how many bottles are there?'
Command parsed: MODE=query, OBJECT=bottle

=== QUERY MODE: Counting bottle ===
  QUERY RESULT: I see 2 bottles
Robot: "I see 2 bottles"
```

## Tips for Speaking

### Good Practices
✅ Say "Simon says" clearly at the beginning
✅ Brief pause after "Simon says" (optional but helps)
✅ Speak at normal conversational pace
✅ Example: "Simon says... search for people"

### What to Avoid
❌ Don't mumble "Simon says"
❌ Don't speak too fast
❌ Don't forget the prefix!

## Testing

### Test Without Robot
```bash
cd /home/georg/Desktop/hands_on_nao/inao
~/.pyenv/versions/robocup_real/bin/python test_voice_recognition.py
```

Try saying:
1. "how many bottles" → Should be rejected
2. "Simon says how many bottles" → Should be accepted
3. "search for people" → Should be rejected
4. "Simon says search for people" → Should be accepted

### Run With Robot
```bash
~/.pyenv/versions/robocup_real/bin/python continuous_vision_test_v4.py
```

## Configuration

The "Simon says" prefix is **mandatory** and cannot be disabled. This is by design to ensure safe and intentional robot control.

If you want to disable it (not recommended), edit `continuous_vision_test_v4.py`:

```python
# In parse_command() method, comment out these lines:
# if not text.startswith("simon says"):
#     print("  -> Ignored (missing 'Simon says' prefix)")
#     return False
# text = text.replace("simon says", "").strip()
```

## Complete Command Reference

| Say This | Robot Does |
|----------|------------|
| Simon says how many bottles | Counts bottles in current view |
| Simon says how many people | Counts people in current view |
| Simon says search for bottles | Rotates head to find bottles |
| Simon says find people | Rotates head to find people |
| Simon says stop | Exits the program |

**Remember**: Every command MUST start with "Simon says"! 🎮
