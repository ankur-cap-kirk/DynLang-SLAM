# Analysis: Cross-Sequence Validation

## Purpose
After 4 rejected hyperparameter experiments (KF / loss / map / init), test
whether the late-drift pattern is sequence-specific or method-specific.

## Result: Method-specific. person_tracking was our best-case sequence.

## Summary

| Sequence | Static | Dynamic | Full | SOTA (DG-SLAM) |
|----------|--------|---------|------|----------------|
| rgbd_bonn_person_tracking | 28.36 | 21.85 | **20.32** | ~4.7 cm |
| rgbd_bonn_balloon         | 69.33 | 57.24 | **57.35** | ~5.0 cm |

Gap to SOTA:
  person_tracking: 4.3x worse
  balloon:         11.5x worse

Gap is NOT constant -> our method is missing a structural component
that SOTA methods have (likely pose graph optimization / loop closure).

## Per-frame trajectory comparison (Full mode)

| Frame | person_tracking | balloon |
|-------|-----------------|---------|
| 10    | 9.1             | 20.6    |
| 20    | 21.6            | 38.7    |
| 30    | 14.1            | 52.2    |
| 50    | 9.8             | 28.8    |
| 60    | 10.3            | 15.4    |
| 70    | 26.0            | 42.6    |
| 80    | 38.6            | 77.6    |
| 90    | 17.4            | 106.7   |
| 99    | 42.6            | 120.7   |

Balloon drifts from frame 10 onwards — fundamentally different failure
pattern from person_tracking's late drift.

## Diagnosis

1. **Balloon has harder camera motion.** Early-frame ATE of 20 cm (vs 9 cm
   for person_tracking) indicates the tracker is failing from the start,
   not at a specific late event.

2. **YOLOv8 doesn't detect balloons** (not in COCO dynamic_classes
   [0,1,2,3,5,14,15,16,17]). Dynamic masking is doing nothing useful
   for the actual dynamic object. Only masks a person who's also in scene.
   Would need motion-based or multi-frame consistency masking to handle.

3. **Language pipeline robust.** 98.6% feature coverage, AE freezes at
   frame 35, all 11 text queries return sensible relevancies, even when
   the pose tracking fails badly. Decoupling of language from pose is
   working.

4. **Dynamic/Full > Static on balloon** (57 < 69). Even with limited
   YOLO classes, masking the person + Bayesian belief decay helps a bit.

## Implications

**Previous tuning was partially sequence-specific.** All 4 experiments
were run only on person_tracking. Hyperparameters that look "baseline"
may be over-fit to person_tracking's specific trajectory shape.

**The true bottleneck is architectural, not hyperparameter.** SOTA
methods that report 4-5 cm on BOTH sequences likely use:
  - Loop closure or pose graph optimization
  - Motion-based dynamic masking (not class-based)
  - Bundle adjustment over keyframe window

Our method has none of these. It's a pure frame-to-model tracker with
local optimization only.

## Honest writeup value

For a CVPR-style course report, this is GOOD data:
- Tested on 2 sequences (shows rigor)
- Identified a specific method limitation (class-based dynamic masking,
  no loop closure)
- Gives concrete future work
- Sets appropriate expectations (not claiming SOTA; demonstrating the
  language+dynamic+GS combination on a realistic dataset)

## Decision

- Revert scripts/test_bonn_full.py SEQUENCE to person_tracking.
- Archive balloon log + analysis.
- Stop tuning; start writing. The research arc is complete:
  * Baseline system built
  * Language + dynamic + Phase 3 improvements (40% ATE reduction)
  * 4 negative tuning experiments (rigor)
  * Cross-sequence validation (reveals method limits)
