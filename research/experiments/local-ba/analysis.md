# Analysis: Local BA over keyframe window (intervention A1)

## Result summary — REJECTED

Pre-registered H1 accept criterion: person_tracking full ATE ≤ **14.00 cm**
(baseline 20.32). All three configurations fail by ≥ 13 cm.

| Config                        | Full ATE | Δ baseline | Verdict |
|-------------------------------|---------:|-----------:|:--------|
| baseline (no Local BA)        |    20.32 |       —    | —       |
| Local BA, no prior            |    27.26 |   **+6.94**| FAIL    |
| Local BA, pose prior=10       |    27.03 |   **+6.71**| FAIL    |
| Local BA, pose prior=100      |    31.42 |  **+11.10**| FAIL    |

H2 (balloon) and H3 (Replica room0) **not run**: H1 primary fails
decisively across three configurations, no point spending GPU budget.

## Per-frame trace — same pattern across all three runs

```
Frame 40: ~9-15 cm   (Local BA is 2-4 cm BETTER than baseline here)
Frame 50: ~9-16 cm   (still better or even)
Frame 60: ~21-29 cm  (drift event starts)
Frame 70: ~41-51 cm  (catastrophic amplification, 17-22 cm single-frame jump)
Frame 80: ~50-57 cm
```

The intervention improves early-trajectory ATE by 2-4 cm but amplifies
the drift event at frame 60-70 by 10+ cm. Baseline's drift event is
~14 cm; Local BA makes it 17-22 cm — *worse*, not corrected.

## Diagnosis — why Local BA cannot fix the drift event in our regime

**On a hard frame where the tracker produces a bad pose**, local BA over
a 5-keyframe window with a single gauge anchor has 4×6 = 24 DOF of
solution space. The photometric + depth loss provides fewer effective
constraints than that (many pixels are textureless, many are masked
dynamic, depth is Kinect-noisy).

The optimizer finds *a* consistent (poses, Gaussians) state — but not
the correct one. Two outcomes compete:

1. **Correct**: pull the bad pose back toward the other 4 poses'
   consistency. Requires a strong signal that the bad pose is wrong.
2. **Wrong**: move the other 4 poses AND the Gaussians to be
   consistent with the bad pose. Looks better to the loss because
   it has 4 pose DOF + millions of Gaussian DOF to play with.

Our regime consistently hits outcome (2) because:

- **Only 1 gauge anchor.** 4 refinable poses vs 1 fixed pose = too few
  witnesses to "the bad pose is the outlier."
- **No ICP anchor.** MonoGS uses frame-to-frame ICP as an independent
  geometric constraint that can't be explained away by Gaussian
  co-adaptation. We don't have one.
- **No global pose graph / loop closure.** Drift doesn't get caught
  anywhere downstream.
- **Small keyframe window (5).** MonoGS uses much larger windows.
- **Sparse map (~16-40k Gaussians).** Not enough constraint density to
  resolve the pose-vs-map ambiguity.

## Why weight tuning doesn't save it

Weight tuning was irrelevant:

- **prior=0**: refinements free to drift (3-4 mm per call in smoke test),
  regression +6.94 cm.
- **prior=10**: refinements shrink to ~1 mm, regression +6.71 cm
  (essentially unchanged).
- **prior=100**: refinements should be ~0.1 mm (near-identity), but
  regression **grows to +11.10 cm**.

The weight=100 result is the key piece of evidence. If the regression
were simply proportional to refinement magnitude, prior=100 should
give near-baseline behavior. Instead it gets *worse* than the
unregularized run. This rules out "refinements are too large" as the
failure cause.

The regression is **structural** — enabling `do_refine` introduces:

- Differentiable `torch.inverse` replacing the numerically-cleaner
  analytic `fast_se3_inverse` for non-anchor poses.
- Every non-anchor pose is rebuilt from `pose_to_matrix(quat, trans)`
  each iteration, introducing small float32 noise vs the original
  matrix.
- Adam state tracking 7-dim pose params (4 quat + 3 trans) × (K-1)
  poses with tiny gradients (near-zero under strong prior) — Adam's
  per-parameter adaptive LR behaves poorly when gradients are
  machine-noise-dominated.
- The writeback to `estimated_poses[kf_id]` subtly changes the
  trajectory used by the tracker's velocity init for the next frame,
  adding more numerical inconsistency.

At prior=100 the refinements are near-zero but the numerical-noise
and Adam-state effects are still fully active — hence the worse
outcome. There's no setting that both disables refinement AND disables
the wiring overhead.

## Structural mismatch with MonoGS

| What MonoGS assumes           | What we have                    |
|-------------------------------|---------------------------------|
| Many keyframes in window      | 5                               |
| Frame-to-frame ICP anchor     | none                            |
| Loop closure / pose graph     | none                            |
| Dense Gaussian map (>100k)    | ~16-40k                         |
| Adam with tuned pose LR       | Adam (poor fit for SE(3))       |

This is the **fourth rejected intervention with the same root pattern:
literature recipe assumed a better upstream or complementary component
than we have** (C: SplaTAM's 200k-init, B: DG-SLAM's DROID-VO upstream,
B': GT-pose diagnostic regime, A1: MonoGS's ICP-anchor + large
keyframe-graph setup).

## Right mechanism for this failure mode

The failure we want to fix is a **discrete single-frame catastrophic
drift event** at frame 60-70. Local BA is architecturally wrong for
this: it's a *sliding-window accumulation of small corrections*,
designed for steady-state drift, not single-frame failures.

The right mechanism is **A2 (tracking-failure detection + retry)** —
multi-hypothesis init at the single bad frame. When the tracker's
confidence is low (loss ratio high, depth residual high), retry
with several pose initializations (velocity-init, previous pose,
small random SE(3) perturbations), keep the one with lowest final
loss. This directly escapes the local min that causes the 14 cm
jump, unlike Local BA which tries to smooth it over multiple frames
after the fact.

A2 is architecturally different from A1 and doesn't share A1's
failure mode:

- Single-frame action, not windowed.
- No gauge ambiguity — only optimizes the current frame pose against
  a fixed map.
- No map co-adaptation — Gaussians don't move during retry.
- Directly addresses the drift-event failure mode identified in
  research-state.yaml.

## Disposition

- Code on `feat/local-ba` branch retained as reference.
- `slam.mapping.refine_poses` flipped to `false` in default config.
- Revert `pose_prior_weight` to 10.0 (less-broken variant if ever
  re-enabled).
- Branch merged to main with `--no-ff` (same discipline as B, C, B').
- `research-state.yaml` updated: A1 rejected, next = A2 retry.

## Budget note

Intervention #4 of the project, #4 rejected. Pattern across C, B, B',
A1 is now decisive evidence that literature-recipe copy-paste doesn't
work in our regime. Remaining interventions (A2 retry, B3 Tukey robust
loss, B4 depth covariance weighting) are explicitly tracker-side
single-frame interventions that don't have this failure mode.
