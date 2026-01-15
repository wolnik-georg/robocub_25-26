# Quick Test Guide - Offline Speech Recognition

## Setup
1. Run the script: `python continuous_vision_test_v4.py`
2. Wait for calibration (4 seconds - **stay silent!**)
3. Look for: `[Listening continuously...]`

## Test Commands (In Order)

### ✅ Test 1: Basic Movement
**Say**: "Simon says wave"
**Expected**: Robot waves, console shows "Command parsed: MODE=movement, ACTION=wave"

### ✅ Test 2: Posture Change  
**Say**: "Simon says sit"
**Expected**: Robot sits down

### ✅ Test 3: Object Query
**Say**: "Simon says how many bottles"
**Expected**: Robot counts bottles in view and responds

### ✅ Test 4: Object Search
**Say**: "Simon says search for person"
**Expected**: Robot rotates head looking for person

### ✅ Test 5: Another Movement
**Say**: "Simon says stand"
**Expected**: Robot stands up

### ✅ Test 6: Stop
**Say**: "Simon says stop"
**Expected**: Program exits

## Success Indicators

### ✓ Working Correctly:
```
Heard: 'simon says wave'
  -> Processing command: 'wave'
Command parsed: MODE=movement, ACTION=wave (matched: 'wave')
```

### ✓ Fuzzy Matching Working:
```
Heard: 'simon says how many bottels'
  -> Processing command: 'how many bottels'
  -> Fuzzy matched 'bottel' to 'bottle'
Command parsed: MODE=query, OBJECT=bottle
```

### ✓ Correctly Ignoring Non-Commands:
```
(No output - background noise ignored silently)
```

### ⚠️ Keyword Without "Simon":
```
[Keyword detected but no 'simon': 'how many bottle']
  -> Ignored (missing 'Simon says' prefix)
```
**Fix**: User needs to say "Simon says" prefix

### ✗ Not Working:
```
(No output at all when speaking)
```
**Possible causes**:
1. Speaking too quietly (increase volume)
2. Energy threshold too high (check calibration value)
3. Microphone not working (check with `arecord -l`)

## Troubleshooting

### No Recognition at All
1. Check energy threshold in console: Should be 800-1200
2. Test microphone: `arecord -d 3 test.wav && aplay test.wav`
3. Speak louder (but not shouting)
4. Reduce background noise

### Too Many False Triggers
1. Energy threshold too low - should auto-adjust to 1.5x ambient
2. Increase manually: Edit line with `energy_threshold = 800` → `1000`

### Wrong Commands Recognized
1. Check if it's fuzzy matching: "bottol" → "bottle" is OK
2. Check keyword list - only keywords are recognized
3. Speak more clearly and slowly

### Continuous Listening Not Working
1. Check console for errors
2. Verify `[Listening continuously...]` message appears
3. Check that thread is daemon (should not block)

## Quick Fixes

### If accuracy is poor:
```python
# Increase energy threshold (line ~347)
self.recognizer.energy_threshold = 1000  # Higher = stricter

# Decrease keyword sensitivity (line ~337)
("simon says", 1e-15),  # More lenient
```

### If missing valid commands:
```python
# Decrease energy threshold
self.recognizer.energy_threshold = 600  # Lower = more sensitive

# Increase keyword sensitivity
("simon says", 1e-25),  # More strict matching
```

### If false triggers on background:
```python
# Increase calibration time (line ~617)
calibration_time = 6 if self.use_offline else 2  # Longer baseline
```

## Performance Targets

| Metric | Target | How to Check |
|--------|--------|--------------|
| Accuracy | >80% | 8/10 commands work correctly |
| False Positives | <10% | <1 false trigger per 10 commands |
| Latency | <2s | Response within 2 seconds of speaking |
| Continuous | ∞ | Works indefinitely without restart |

## Example Session (Good Performance)

```
[Listening continuously...]

Heard: 'simon says wave'
  -> Processing command: 'wave'
Command parsed: MODE=movement, ACTION=wave (matched: 'wave')
=== MOVEMENT MODE: wave ===

Heard: 'simon says sit down'
  -> Processing command: 'sit down'
Command parsed: MODE=movement, ACTION=sit (matched: 'sit down')
=== MOVEMENT MODE: sit ===

Heard: 'simon says how many cups'
  -> Processing command: 'how many cups'
Command parsed: MODE=query, OBJECT=cup
=== QUERY MODE: Counting cup ===
QUERY RESULT: I see 2 cups

Heard: 'simon says stand'
  -> Processing command: 'stand'
Command parsed: MODE=movement, ACTION=stand (matched: 'stand')
=== MOVEMENT MODE: stand ===

Heard: 'simon says stop'
  -> Processing command: 'stop'
Command parsed: STOP
=== STOP command received ===
```

**Result**: 5/5 commands = 100% accuracy ✓

## Common Test Mistakes

### ❌ Speaking Too Fast
"SimonsayswaveSimonsaysit" - Run words together
**Fix**: Pause between commands, speak at normal pace

### ❌ Incomplete Commands
"Simon wave" - Missing "says"
"Says wave" - Missing "Simon"
**Fix**: Always say full "Simon says ..." 

### ❌ Not Waiting for Calibration
Starting to speak during "Calibrating..." message
**Fix**: Wait for "[Listening continuously...]" before speaking

### ❌ Using Non-Keywords
"Simon says container" instead of "bottle"
**Fix**: Use keywords from list (bottle, cup, person, chair, etc.)

### ❌ Background Interference
TV/music playing during test
**Fix**: Turn off background noise sources

## Advanced Testing

### Test Fuzzy Matching:
- "Simon says how many bottels" → should match "bottle"
- "Simon says mov forward" → should match "move forward"
- "Simon says waiv" → should match "wave"

### Test Continuous Operation:
1. Say 10 commands in a row
2. All should work without restart
3. No degradation over time

### Test Noise Resistance:
1. Start with quiet room (baseline)
2. Add moderate noise (music at low volume)
3. Commands should still work (but maybe 70% instead of 90%)

### Test Object Recognition:
Place objects in view:
- "Simon says how many bottles" (with bottles visible)
- "Simon says search for person" (with person in frame)
- Should get accurate counts

## Benchmark Results

Record your results:
```
Date: _______
Environment: Quiet room / Moderate noise / Loud
Commands attempted: ___
Commands successful: ___
False positives: ___
Accuracy: ____%
Notes: ________________________
```

Target: **>80% accuracy in quiet room**

---

Good luck! 🎤🤖
