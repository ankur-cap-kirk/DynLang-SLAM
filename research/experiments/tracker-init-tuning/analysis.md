# Analysis: Tracker Pose-Init (4A)

## Result: Null. Reverted to baseline A.

## Summary (BONN person_tracking, 100 frames)

| Config | Static | Dynamic | Full | Time/frame (s) | Gaussians | Extractions |
|--------|--------|---------|------|----------------|-----------|-------------|
| A (velocity init)   | 28.36 | 21.85 | 20.32 | 1.00-1.49 | 33603 | 7 |
| 4A (identity init)  | 23.49 | 24.01 | 23.81 | 0.37-0.62 | 18030 | 3 |

## Key observations

1. **Runtime dropped 2.4x** but accuracy tied (within variance).

2. **Gaussian count frozen at ~18k** (baseline reaches ~33k).
   `_expand_map` didn't trigger because identity init underestimates motion.

3. **Only 3 KFs extracted** vs baseline's 7.
   Motion-based keyframe thresholds (trans=5cm, rot=5deg) didn't fire because
   the tracker returned poses close to the starting pose when hard frames
   early-stopped the optimizer at a plateau.

4. **Same frame-60 to frame-70 ATE jump** (17.58 -> 31.96 cm).
   The specific drift event is UNAFFECTED by pose init strategy.

## Interpretation

Identity-relative init doesn't fix the drift. It makes the tracker silently
underestimate motion on hard frames (the plateau-stop returns near the
starting pose), which:
- Reduces map expansion work (faster)
- Reduces keyframe count
- Does NOT reduce final ATE

Velocity init actually DOES something useful when motion is smooth: it gives
the optimizer a better basin to descend into. The failure isn't at init —
it's at the optimizer itself, which can't recover once hit with a hard frame.

## Meta-signal (10 configs, 4 experiments)

```
Experiment 1 (KF-density):      B, D         -> rejected
Experiment 2 (Loss-weight):     E, F, G      -> rejected
Experiment 3 (Map-growth):      H, I, J      -> rejected
Experiment 4A (Pose-init):      4A           -> tie (with side effects)
```

The late-drift at frame 65-70 is a DISCRETE EVENT, not a tunable
accumulation. All 10 configs produce the same trajectory shape:
  - ATE < 15 cm through frame 60
  - Sudden ~14 cm jump around frame 65-70
  - Continued growth to 40-55 cm by frame 99

This is consistent with a specific hard frame (sharp rotation, motion
blur, occlusion peak) that the current tracker cannot handle, regardless
of hyperparameter tuning.

## Next candidates (ranked by expected impact)

1. **Validate on another BONN sequence.** Run rgbd_bonn_balloon or
   rgbd_bonn_crowd to test if this failure is method-specific or
   sequence-specific. HIGH expected information value.

2. **Visualize the failure event.** Render frames 60-75 to see what's
   actually happening in the video. Identifies the failure mode.

3. **Structural fix: recovery mechanism.**
   - Detect tracking failure (loss spike, pose jump)
   - On failure: retry with multiple init candidates, pick best loss
   - Or: add loop closure over keyframe window

4. **Optimizer schedule (4B).** Warmup + decay. Low expected impact
   given Exp 1-3 all rejected.

## Decision
Revert to baseline A. Recommend Option 1 (validate on another sequence)
as highest-value next step before further algorithmic changes.
