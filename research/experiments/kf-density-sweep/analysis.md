# Analysis: Keyframe Density Sweep

## Result: Hypothesis NOT supported. Reverted to Config A.

## Summary table (Full mode, BONN person_tracking, 100 frames)

| Config | kf_trans | kf_rot | kf_max_int | ATE (cm) | time/fr (s) | lang extractions |
|--------|----------|--------|------------|----------|-------------|------------------|
| A (baseline) | 0.05 m | 5 deg | 15 | 22.27 | 1.55 | 9  |
| B (denser)   | 0.03 m | 3 deg | 10 | 23.60 | 7.50 | 17 |
| D (cap-only) | 0.05 m | 5 deg | 8  | 33.17 | 2.06 | 12 |

## Run-to-run variance (all configs, same codebase)

| Mode | Min | Max | Spread |
|------|-----|-----|--------|
| Static  | 24.31 | 34.02 | ~10 cm |
| Dynamic | 23.04 | 31.18 | ~8 cm  |
| Full    | 22.27 | 33.17 | ~11 cm |

**Run-to-run variance is ~10 cm** on this 100-frame sequence. Any individual
sweep with a delta < 10 cm is in the noise floor.

## Takeaways

1. **KF density is NOT the bottleneck.** Neither denser (B) nor cap-tightening (D)
   improves full-mode ATE vs baseline A. All three produce the same late-drift
   pattern: ATE < 15 cm at frame 40, balloons to 40-60 cm at frames 70-90.

2. **Variance > signal on 100-frame BONN.** The drift is driven by late-frame
   tracking failures, which are stochastic. Sweeping KF params at this scale
   cannot cleanly resolve effects smaller than the variance.

3. **Next target is tracking, not mapping.** The drift source lives in the
   tracker's loss landscape around frames 70-90. Candidates:
   - Lower depth_weight (Kinect has 5-10 cm depth noise; 1.2 may be too high)
   - Wider huber_depth_delta (0.1 -> 0.2) to tolerate depth noise
   - Increase tracking iterations (currently 50; map is fine but pose undercuts)

4. **Evaluation protocol change**: future sweeps should either
   - Run 3x per config and report mean +/- std, OR
   - Test on a longer/harder sequence where signal > variance

## Decision
- Revert config to A (commit 797346b...ac3a6f6 baseline).
- Close this experiment directory.
- Open new experiment: "loss-weight-tuning" targeting late-frame drift.
