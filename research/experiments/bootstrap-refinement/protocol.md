# Experiment: Bootstrap-Window Joint Pose Refinement (intervention BR1)

## Motivation

Under the corrected Umeyama-aligned ATE metric (see
`research-state.yaml:evaluation_correction_2026_04_23`), DynLang-SLAM
measures:

| scene              | aligned ATE (cm) | raw ATE (cm) | ratio |
|--------------------|-----------------:|-------------:|------:|
| H1 person_tracking |             7.56 |        29.16 |  3.86x |
| H2 balloon         |             8.55 |        57.89 |  6.77x |
| H3 replica_room0   |             3.68 |        13.02 |  3.54x |

The **raw/aligned ratio** measures what fraction of total ATE is a
rigid-body offset (absorbed by SE(3) alignment) vs per-frame jitter
(preserved by alignment). A 3.5–6.8x ratio means our residual error is
**dominated by rigid bias**, not noise.

**Diagnosis**: velocity-init during the first ~10-20 frames commits pose
estimates against a sparse, freshly-seeded Gaussian map. Those early
poses accumulate a rigid offset; every subsequent velocity-extrapolated
pose inherits it. Mapping-side masks (B depth-warp, B' PIPs) and
tracker-side retry (A2) all failed — they don't touch the early
poses that are the root cause. And continuous local BA (A1, rejected
under raw ATE) refines only the most-recent 4 keyframes per mapping
call, not the early ones where the bias lives.

## Hypothesis under test

> Retroactively re-optimizing the first N_bootstrap frames' poses
> **once**, after the map has had time to stabilize, will remove the
> rigid offset and collapse the raw/aligned ratio — yielding a larger
> aligned-ATE improvement per unit compute than any prior intervention.

Key distinction from already-rejected A1:
- A1: refines last 4 KFs of a 5-KF window, **every mapping call**,
  forever. Early-frame bias never gets touched because by the time
  A1 fires on later windows, the anchor rolls forward past the
  bootstrap region.
- **BR1**: refines ALL KFs collected in [0, N_bootstrap],
  **exactly once**, with frame-0's GT pose as gauge anchor. Targets
  the bootstrap region specifically.

## Algorithm

At frame `N_bootstrap` (after normal `process_frame` completes):

1. Collect all keyframes accumulated so far: `self.keyframes[0..K-1]`.
2. Anchor `poses[0]` (frame 0 GT pose) as gauge. Promote `poses[1..K-1]`
   to learnable (quat, trans) parameters, initialized from the tracker's
   committed estimates.
3. Jointly optimize Gaussian parameters AND non-anchor poses for
   `N_iters = 200` iterations of Adam using the normal
   photometric+depth+SSIM loss, averaged across all K frames.
4. Add MonoGS-style pose prior (weight=10.0) pulling refinement toward
   the tracker's initial estimate. Prevents under-constrained poses from
   sliding into bad minima.
5. Write refined poses back to:
   - `self.keyframes[i]["pose"]` for i in [1..K-1]
   - `self.estimated_poses[frame_id]` for each refined keyframe

Non-keyframe poses (between keyframes) are **not** retroactively
refined — they inherit the rigid correction applied to the flanking
keyframes via their velocity-init lineage. A follow-up BR2 could
interpolate, but the hypothesis under BR1 is that bootstrap KF
correction alone is enough.

After the one-shot bootstrap event, the pipeline continues in its
normal mode. `refine_poses=False` stays as the mapper default; BR1 does
not introduce continuous BA (that was A1 and was rejected).

## Parameter choices (fixed, no tuning allowed under protocol)

| Parameter             | Value   | Rationale |
|-----------------------|---------|-----------|
| `N_bootstrap` frames  | 15      | ~half of the typical SLAM bootstrap length; at BONN stride=2 = 30 raw frames ≈ 1.5s |
| Refinement iters      | 200     | ~3.3x single mapping call; enough for pose-prior-regularized optimum |
| `lr_pose_trans`       | 1.0e-4  | same as A1 (conservative; tracker already has rough estimate) |
| `lr_pose_quat`        | 5.0e-4  | same as A1 |
| `pose_prior_weight`   | 10.0    | same as A1 default; A1 analysis found prior=100 strictly worse |
| Gauge anchor          | frame 0 | only frame with GT pose seed; all others are tracker output |

## Accept criteria (pre-registered against ALIGNED ATE)

Baselines (from `aligned_baselines_2026_04_23`):
- H1 person_tracking: **7.56 cm**
- H2 balloon:         **8.55 cm**
- H3 replica_room0:   **3.68 cm**

| Sequence         | Baseline | Accept if aligned ATE ≤ | Hypothesis |
|------------------|---------:|------------------------:|:-----------|
| person_tracking  |     7.56 |             **5.50**    | H1 primary |
| balloon          |     8.55 |             **6.00**    | H2 primary |
| replica room0    |     3.68 |             **4.10**    | H3 safety  |

- H1 target: **−2.06 cm (27% reduction)** — puts us within 0.77 cm of
  DG-SLAM's 4.73 cm on BONN.
- H2 target: **−2.55 cm (30% reduction)** — balloon has the highest
  ratio (6.77x), so proportionally more headroom if the bias-hypothesis
  is correct; it should improve more than H1, not less.
- H3 budget: **+0.42 cm regression tolerated (11%)** — static scene,
  bootstrap bias should be small, so regression should be minimal.
  Generous safety margin accounts for stochastic variation.

### Decision matrix

**ACCEPT** if:
- H1 ≤ 5.50 cm, AND
- H2 ≤ 6.00 cm, AND
- H3 ≤ 4.10 cm

**REJECT** if ANY of:
- H3 > 4.10 cm (safety regression on static scene)
- H1 improves by less than 0.5 cm (no real signal)
- Both H1 AND H2 regress

**Partial accept** (H1 improves, H2 doesn't):
- Document as "bootstrap refinement harvests person-scene bias but
  balloon regime dominated by dynamic-object residual, not bootstrap
  offset" — and keep feature gated by config.

## Why this should survive where A1 did not

| A1 failure mode                               | BR1 response                                                                 |
|-----------------------------------------------|------------------------------------------------------------------------------|
| Continuous per-window BA couples to density schedule | BR1 fires once; no continuous interaction with densification |
| 4-KF window anchor rolls past bootstrap region | BR1 anchor is frame 0 specifically — the one pose we know is exact |
| Gaussian init bias propagates via velocity-init before A1 can catch it | BR1 runs AFTER 15 frames of map growth; map has geometric evidence of later-frame structure to correct earlier poses |
| Evaluated on raw ATE (biased toward bias-removal) | BR1 pre-registered on **aligned** ATE, which penalizes jitter introduction and rewards only genuine rigid-offset correction |

## Verification protocol

Script: `scripts/br1_eval.py`

1. **H1 person_tracking**: 100 frames stride=2. Report aligned ATE
   with and without BR1. Log pose deltas per refined keyframe.
2. **H2 balloon**: 100 frames stride=2. Report aligned ATE.
3. **H3 replica room0**: 100 frames, full res. Report aligned ATE.
4. **Safety smoke check**: print max pose delta across refined KFs; if
   any single delta > 20 cm translation or > 10 deg rotation, log a
   warning — indicates optimization landed in a bad basin.

Runtime budget: 200-iter bootstrap refinement + Gaussian co-optimization
should take 15-30 s extra on each sequence. Acceptable.

## What this intervention is NOT doing

- NOT running continuous local BA — that is A1, already rejected.
- NOT refining non-keyframe poses — could be BR2 if BR1 succeeds.
- NOT changing the steady-state tracker or mapper — BR1 is an
  additive one-shot event gated by config flag.
- NOT using GT poses beyond frame 0 — the anchor is the same GT seed
  the pipeline already uses.
- NOT touching the raw ATE baselines — this experiment is scored
  **entirely** under the aligned metric.

## Pre-registration signed (2026-04-23)

Claude (assistant agent for DynLang-SLAM research).
Budgets above are binding. Post-hoc threshold tuning = protocol
violation. Documented under pre-commit; results filed in
`analysis.md` after experiment completes.
