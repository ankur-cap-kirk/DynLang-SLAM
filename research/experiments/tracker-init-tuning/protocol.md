# Experiment: Tracker Pose-Init (Config 4A)

## Research Question
Is constant-velocity pose init responsible for late-frame drift on BONN?

## Hypothesis
After three rejected experiments (KF-density, loss-weight, map-growth), the
late-drift pattern is invariant to anything in the mapper/losses. The drift
must originate in the tracker.

Constant-velocity init extrapolates:
  init_pose[t] = prev @ (prev2^-1 @ prev)

This is excellent when camera motion is smooth (Replica), but fails when
camera changes direction or accelerates (common in handheld BONN sequences).
If v(t-1) points the wrong way relative to v(t), velocity init lands the
tracker in a bad basin that gradient descent can't escape.

Identity-relative init (just reuse prev pose) starts at a neutral point.
It has lower expected accuracy when motion IS smooth, but doesn't
catastrophically fail on direction changes.

Hypothesis: on BONN person_tracking, identity-relative init reduces the
late-drift balloon at frames 70-99.

## Locked Before Results
- Dataset: BONN rgbd_bonn_person_tracking, 100 frames, stride=2
- Test script: scripts/test_bonn_full.py
- Mode: Full (dynamic + language) — primary
- Metric: ATE RMSE (cm) on full trajectory

## Configurations
| Config | use_velocity_init | Notes |
|--------|-------------------|-------|
| A (baseline) | true  | current default |
| 4A           | false | identity-relative init |

## Prediction
- 4A Full ATE: 15-22 cm (wide range, reflects uncertainty)
- If 4A matches baseline within variance: velocity init isn't the issue
- If 4A beats baseline by >4 cm: velocity init IS the issue, investigate 4B/4C
- If 4A regresses significantly: velocity init is actively helping,
  bottleneck is different (consider optimizer landscape, loop closure)

## Decision Rule
Must beat A-rebaseline (20.32 cm) by >4 cm to commit as new default.
Otherwise document result and move to 4B (optimizer schedule).
