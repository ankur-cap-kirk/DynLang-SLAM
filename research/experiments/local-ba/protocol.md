# Experiment: Local Bundle Adjustment over keyframe window (intervention A1)

## Motivation

All three prior architectural interventions (C silhouette-gate, B
depth-warp mask, B' CoTracker mask) were rejected for the same
structural reason: they operate on the MAPPING side while the actual
drift event happens on the TRACKER side. The tracker commits a bad
single-frame pose; every downstream Gaussian is seeded with that wrong
pose; no mapping-side fix can undo it.

Verified in the current codebase (`dynlang_slam/slam/mapper.py:65-76`):

```python
def map(
    self,
    gaussian_map: GaussianMap,
    frames: list[dict],
    poses: list[torch.Tensor],   # <-- plain tensors, not nn.Parameter
    ...
```

Poses are **frozen** during mapping. The mapper optimizes Gaussians
given fixed poses. No cross-keyframe pose refinement, no way to
retroactively correct a bad tracker output.

This is the single biggest missing mechanism between our pipeline and
every SOTA 3DGS-SLAM system (MonoGS, SplaTAM, Gaussian-SLAM,
BDGS-SLAM). The project roadmap has flagged this as NEXT since the
intervention plan was written.

## What the change does

**Algorithm (MonoGS-style local BA):**

1. In `Mapper.map()`, promote the last K-1 keyframe poses in the window
   to learnable parameters (quaternion + translation). The OLDEST
   keyframe in the window stays fixed as **gauge anchor** — without
   this, the whole trajectory can translate/rotate freely under the
   photometric+depth loss (gauge ambiguity).
2. Add two pose parameter groups to the Adam optimizer:
   - `lr_pose_trans = 1e-4` (conservative; tracker already has a rough
     estimate, BA only polishes)
   - `lr_pose_quat = 5e-4`
3. Each iteration: rebuild the 4x4 pose matrix from (normalized quat,
   trans) for each non-anchor keyframe, use it in the viewmat for
   rendering. Gradients flow back through the renderer into the pose
   params.
4. After optimization, write the refined poses back to:
   - `self.keyframes[i]["pose"]` — so future mapping windows see the
     corrected history
   - `self.estimated_poses[kf_id]` — so ATE computation sees the
     corrected trajectory

**Parameterization:**

- Each non-anchor keyframe: `(quat: 4, trans: 3)`, initialized from the
  tracker's output pose.
- Quaternion normalized on each iteration (projection back to S^3).
- This is a minimal parameterization with implicit manifold constraint.
  Simpler than SE(3) Lie algebra and matches the convention already
  used for Gaussian orientations.

**Strictly additive design:**

- New flag `refine_poses: bool` on `Mapper.__init__()`, defaulting to
  the config value. Setting `lr_pose_trans = lr_pose_quat = 0` OR
  `refine_poses = False` recovers exact prior behavior.
- Non-keyframe poses (between keyframes) are NOT refined in this
  intervention. Addressed if needed in a later pass.

## Hypothesis

- **H1 (primary, person_tracking)**: baseline full 20.32 cm. Local BA
  should retroactively smooth the known drift-event at frame 65-70.
  Accept if full ATE ≤ **14.00 cm** (30% improvement, matches MonoGS's
  reported Replica gains).
- **H2 (primary, balloon)**: baseline full 57.35 cm. The balloon
  regime has tracker drifting on dynamic pixels; BA using later
  keyframes as evidence should correct some of that. Accept if full
  ATE ≤ **40.00 cm** (same threshold as B/B' used for balloon).
- **H3 (safety, Replica room0)**: baseline full 0.72 cm. A well-posed
  static scene should not regress under local BA. Accept if full
  ATE ≤ **1.50 cm** (generous safety margin for stochastic variation).

## Why this intervention should survive where B/B'/C did not

| Prior rejection | Its failure mode | Why local BA doesn't share it |
|---|---|---|
| C silhouette-gate | sparse-map regime collapses alpha gradient | we don't gate at all |
| B depth-warp | velocity-init pose error dominates residual | we use TRACKED pose as init, refine from there |
| B' PIPs mask | map-side mask cannot undo tracker drift | we DIRECTLY refine the pose, not the mask |

Local BA is **strictly additive with safe fallback**: if the refined
pose is worse than the tracker's initial pose, the photometric loss
goes up, gradients push the pose param back toward the initial basin.
Unlike B/B'/C, it cannot catastrophically over-mask or collapse
gradients.

## Implementation plan

1. New utility `dynlang_slam/utils/se3.py`: `pose_to_quat_trans(T)` and
   `quat_trans_to_pose(q, t)` with quat normalization.
2. `Mapper.__init__` takes `lr_pose_trans`, `lr_pose_quat`,
   `refine_poses` flags.
3. `Mapper.map()`:
   - If `refine_poses` and `len(poses) > 1`: build param list from
     poses[1:], add to Adam, use rebuilt poses in the inner loop.
   - Return `refined_poses` in info dict (anchor + param-derived).
4. `SLAMPipeline`: after mapping, if `refined_poses` is present in
   info, write back to `keyframes[-K:][j]["pose"]` and
   `estimated_poses[keyframes[j]["frame_id"]]` for j in 1..K-1.
5. Config `slam.mapping.refine_poses: true`,
   `lr_pose_trans: 1.0e-4`, `lr_pose_quat: 5.0e-4`.

## Verification

1. **Smoke test**: on Replica room0 first 20 frames, pose params should
   barely move (static scene, tracker already ~0.7 cm). Log per-frame
   `||Δpose||`. Expect < 1 mm / < 0.01 rad.
2. **person_tracking full-run**: compare to baseline 20.32 cm. Also
   verify per-frame ATE trace shows the frame-65-70 event is smaller.
3. **balloon full-run**: compare to baseline 57.35 cm.
4. **Replica room0 full-run**: H3 safety check.

## Accept criteria (pre-registered)

| Sequence         | Baseline | Accept if full ATE ≤ | Hypothesis |
|------------------|---------:|---------------------:|:-----------|
| person_tracking  |    20.32 |        **14.00**     | H1 primary |
| balloon          |    57.35 |        **40.00**     | H2 primary |
| replica room0    |     0.72 |         **1.50**     | H3 safety  |

Accept the intervention if H1 AND H3 pass. H2 is a stretch target —
if H2 fails but H1+H3 pass, accept with "addresses drift-event failure
mode on person_tracking; balloon remains open for next intervention."

Reject if H3 fails (safety regression) OR H1 fails (the primary
drift-event story doesn't hold).

## Runtime budget

Per-iteration cost: 1 pose param update × 4 keyframes × 60 iters ×
negligible-FLOPs (pose is 7 floats) ≈ free. Expected overhead < 5%
per mapping call.

## What I am explicitly NOT doing

- NOT refining non-keyframe poses (defer to A2 retry).
- NOT adding loop closure / global pose graph (defer to later if
  needed).
- NOT changing the tracker's optimizer (defer to B2/B3 if needed).
- NOT initializing pose params from anywhere except the tracker's
  output (velocity init stays as-is upstream).

The question under test: **does cross-keyframe pose refinement, with a
fixed gauge anchor and small pose LR, retroactively correct the
drift-event that every mapping-side intervention has failed to reach?**
