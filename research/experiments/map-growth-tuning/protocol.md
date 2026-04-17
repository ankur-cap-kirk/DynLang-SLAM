# Experiment: Map-Growth Tuning

## Research Question
Does accelerating map growth during exploration reduce late-frame drift on
BONN person_tracking?

## Hypothesis
After two negative experiments (KF-density and loss-weight), the strongest
remaining signal is: in all configurations, Gaussian count correlates with
ATE quality, AND late-frame drift (frames 70-90) is unchanged by tracker-side
or loss-side fixes.

Theory: when the camera enters new regions at frame 70+, `_expand_map` adds
new Gaussians but these need multiple mapping iterations to stabilize (pose,
color, opacity converge slowly). Meanwhile the tracker tries to align to a
stale/sparse map and lands in bad basins, propagating drift.

Three independent levers target map-growth speed:
  1. More mapping iterations (let new Gaussians converge before tracker uses them)
  2. Lower new-Gaussian silhouette threshold (add Gaussians more aggressively)
  3. More frequent densification (adaptive split/clone responds faster)

## Locked Before Results
- Dataset: BONN rgbd_bonn_person_tracking, 100 frames, stride=2
- Test script: scripts/test_bonn_full.py
- Mode: Full (dynamic + language) — primary metric
- Metric: ATE RMSE (cm) on full trajectory

## Configurations
| Config | mapping.iterations | new_gaussian_thresh | densify_interval |
|--------|--------------------|---------------------|------------------|
| A (baseline) | 60  | 0.5 | 10 |
| H (more iters)   | 100 | 0.5 | 10 |
| I (lower thresh) | 60  | 0.3 | 10 |
| J (densify fast) | 60  | 0.5 | 5  |

## Predictions
- H: 15-20 cm, +40% mapping time, safest bet
- I: 16-25 cm, more Gaussians, risk of over-population and instability
- J: 15-22 cm, unpredictable (tracker may like or hate fast-changing map)

## Decision Rule
- Must beat A-rebaseline (20.32 cm) by > 4 cm (roughly 2x variance) to count as a real win
- If multiple win, prefer lowest compute cost
- If all three regress, pivot to pose-init experiments (constant-velocity reconsideration)

## Variance reference
Prior sessions measured ~3 cm run-to-run variance for baseline A.
