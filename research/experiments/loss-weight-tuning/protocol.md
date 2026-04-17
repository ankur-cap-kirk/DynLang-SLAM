# Experiment: Loss-Weight Tuning for BONN Kinect Noise

## Research Question
Does rebalancing RGB/depth loss weights (and widening Huber delta) reduce
late-frame drift on BONN person_tracking under Kinect v2 depth noise?

## Hypothesis
Current defaults (depth_weight=1.2, rgb_weight=0.5, huber_depth_delta=0.1) were
tuned for Replica's synthetic clean depth. Kinect v2 at 3-4 m has 5-10 cm noise,
which becomes the dominant gradient signal in tracking. This explains:
- Late-drift pattern (frames 70-90) that persisted across ALL KF configs
- Run-to-run variance of ~10 cm (stochastic depth-noise-driven pose wander)

Lowering depth_weight and/or widening huber_depth_delta should let photometric
signal dominate when depth becomes unreliable.

## Locked Before Results
- Dataset: BONN rgbd_bonn_person_tracking, 100 frames, stride=2
- Test script: scripts/test_bonn_full.py
- Mode: Full (dynamic + language)
- Metric: ATE RMSE (cm) on full trajectory

## Configurations
| Config | depth_w | rgb_w | huber_depth_delta | Rationale |
|--------|---------|-------|-------------------|-----------|
| A-rebaseline | 1.2 | 0.5 | 0.10 | re-run baseline to measure variance |
| E (low-depth)   | 0.6 | 0.5 | 0.10 | halve depth weight |
| F (wide-Huber)  | 1.2 | 0.5 | 0.20 | same weight, more tolerance |
| G (combined)    | 0.8 | 0.7 | 0.15 | rebalance + moderate Huber |

## Predictions
- A-rebaseline: 15-32 cm (shows variance envelope)
- E: 15-20 cm ATE (main hypothesis: depth over-weighted)
- F: 18-24 cm ATE (partial fix, Huber widens tolerance band)
- G: 12-18 cm ATE (combined — best case but highest risk of instability)

## Decision Rule
- Pick config with lowest ATE RMSE
- Must beat A-rebaseline by > 5 cm (half the measured variance spread) to count as a real win
- If tied: prefer config that also improves per-frame trajectory smoothness
