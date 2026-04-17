# Analysis: Map-Growth Tuning

## Result: Hypothesis NOT supported. Reverted to baseline A.

## Summary (BONN person_tracking, 100 frames, Full mode is primary)

| Config | mapping.iter | new_gauss_thresh | densify_int | Static | Dyn | Full |
|--------|--------------|------------------|-------------|--------|-----|------|
| A-rebaseline | 60  | 0.5 | 10 | 28.36 | 21.85 | **20.32** |
| H (more iters)   | 100 | 0.5 | 10 | 29.56 | 33.51 | 27.98 |
| I (lower thresh) | 60  | 0.3 | 10 | 34.75 | 23.41 | 20.01 |
| J (densify fast) | 60  | 0.5 | 5  | 31.10 | 21.61 | 24.31 |

## Key observations

1. **H doubled mapping compute for +7.66 cm regression in full mode.**
   More iterations stretched the LR decay schedule, raising effective LR at
   former termination point. Over-optimization of each KF biases the map
   toward frame-specific geometry, hurting tracking stability.

2. **I was the only tie — slightly different trajectory, same RMSE.**
   Lower silhouette threshold triggered more aggressive growth AND more
   aggressive contamination cleanup (8635 Gaussians removed at frame 90
   vs typical 2-4k). Net zero.

3. **J regressed modestly.** More frequent densification (every 5 mapping
   calls instead of 10) didn't keep pace with map coverage needs; instead
   it induced more split/clone overhead without helping tracker.

4. **Late-drift pattern is invariant across all 9 configs tested.** Across
   three independent experiments (KF-density: B, D; loss-weight: E, F, G;
   map-growth: H, I, J), ATE always balloons at frames 70-99 regardless
   of what we tune on mapping/losses side.

## Revised theory (third iteration)

After rejecting KF-density (Exp 1), loss-weighting (Exp 2), and map-growth
(Exp 3), the bottleneck cannot be on the mapping side. The tracker is
failing around frame 70 regardless of map quality.

**Candidate root causes (for next experiments):**

1. **Constant-velocity pose init fails at direction changes.**
   `use_velocity_init: true` extrapolates v(t-1) -> t. When camera
   changes direction (turn, stop-and-go), this lands tracker in a
   bad basin. Prior negative result: coarse-to-fine tracking
   (already disabled due to regression) points same direction.

2. **Optimizer too local.**
   Current lr_pose=0.002, cosine decay to 0.05x, 50 iterations.
   Once initial pose is off, gradient descent stays in wrong basin.
   Could try: warmup (high LR for first 5 iters) then decay, OR
   restart-on-plateau (if loss doesn't drop in 5 iters, jump to
   identity-relative init and retry).

3. **No recovery mechanism.** Pure frame-to-model tracking has no way
   to detect or undo a drifted pose. Loop closure or periodic
   global BA on keyframe window could re-anchor.

## Next experiment candidates

**Experiment 4A: Pose-init reset.** Disable use_velocity_init, see if
identity-relative init performs better on this sequence. Cheap.

**Experiment 4B: Optimizer schedule.** Try warmup+decay (5 iters at
2x baseline LR, then cosine decay). Cheap.

**Experiment 4C: Global BA.** Periodic bundle adjustment over recent
keyframe window (expensive, architectural).

## Decision

- Revert config to A (mapping.iters=60, new_gauss_thresh=0.5, densify_int=10).
- Close this experiment directory.
- Open Experiment 4: tracker-side fixes (pose init + optimizer schedule).

## Meta-observation on tuning discipline

Three negative experiments in a row is a **strong meta-signal**: we're
tuning the wrong subsystem. The pattern is that ATE < 15 cm is achievable
(first 40 frames consistently hit this), but something at frame 70+
breaks and propagates. Further sweeping on map/loss parameters would
waste compute. Must move to tracker internals.
