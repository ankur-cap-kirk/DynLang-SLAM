# Experiment: Keyframe Density Sweep

## Research Question
Does increasing keyframe density reduce late-frame ATE drift on BONN person_tracking?

## Hypothesis
Current motion-based KF selection (Phase 3 baseline A: trans=0.05m, rot=5°, max=15)
produces sparse keyframes. The map becomes stale by frame 60+, causing ATE to balloon
from ~10cm (early) to ~40cm (late). Denser KFs -> fresher local map -> less drift.

## Locked Before Results
- Dataset: BONN rgbd_bonn_person_tracking, 100 frames, stride=2
- Test script: scripts/test_bonn_full.py
- Mode: Full (dynamic + language)
- Metric: ATE RMSE (cm) on full trajectory

## Configurations
| Config | kf_trans_thresh | kf_rot_thresh | kf_max_interval |
|--------|-----------------|---------------|-----------------|
| A (baseline) | 0.05 | 0.087 (5 deg) | 15 |
| B (denser)   | 0.03 | 0.052 (3 deg) | 10 |
| C (densest)  | 0.02 | 0.035 (2 deg) | 8  |
| D (cap only) | 0.05 | 0.087 (5 deg) | 8  |

## Predictions
- B: 15-18 cm ATE, ~65 KFs, +20% runtime
- C: 12-16 cm ATE, ~85 KFs, +40% runtime
- D: 16-20 cm ATE, ~55 KFs, +10% runtime

## Decision Rule
- Pick config with lowest ATE RMSE AND runtime < 3 s/frame
- If ATE ties within 2 cm, prefer faster config
