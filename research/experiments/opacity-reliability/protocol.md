# Intervention D1: Opacity reliability mask + loss weight flip (DG-SLAM-style)

## Motivation

Code comparison against DG-SLAM (NeurIPS 2024, arXiv 2411.08373) revealed
two targeted deltas in the tracker loss formulation:

### Delta 1 — RGB mask asymmetry

Our `dynlang_slam/core/losses.py::compute_losses` already uses accumulated
opacity `alpha` for reliability weighting, but **asymmetrically**:

- **Depth loss**: hard threshold `(gt_depth > 0) & (alpha > 0.5)` ✓
  matches DG-SLAM's `Ôᵢ = 1 iff accumulated_opacity > τ_track`.
- **RGB loss**: soft linear weighting `rgb_mask = alpha`. A pixel with
  `alpha = 0.1` (under-mapped) still contributes 10% of its residual.

DG-SLAM applies the hard threshold to **both** photometric and depth
terms. The concern: Adam's per-parameter adaptive learning rate can be
pulled by small-but-consistent gradients from unreliable (under-mapped)
pixels. On our sparse map (16–40k Gaussians), a large fraction of the
image can have `alpha < 0.5` — exactly the regime where soft weighting
fails.

### Delta 2 — Loss weight inversion

| | λ_rgb | λ_depth | λ_ssim | rgb/depth ratio |
|---|---|---|---|---|
| DG-SLAM (paper Sec 4.2)     | 0.9 | 0.1 | 0.2 | 9.0  |
| Our default (configs/default.yaml) | 0.5 | 1.2 | 0.2 | 0.42 |

We are depth-dominant by >20× relative weight. On BONN with Kinect
depth noise, this amplifies sensor error. DG-SLAM is photometric-
dominant. Worth an ablation.

## Hypothesis (pre-registered)

**H0 (wiring)**: After the change, compute_losses still produces finite
gradients; smoke test 25 frames on BONN person_tracking runs without
NaN or divergence. Final ATE within ±20% of baseline on the smoke run.

**H1 (primary)**: BONN person_tracking, 100 frames stride=2, full
pipeline (dynamic + language).
- Accept if ATE RMSE ≤ **14.00 cm** (≥30% improvement over 20.32 baseline).
- Marginal-accept if 14.00 < ATE ≤ 18.00 cm (need H2 to confirm).
- Reject if ATE > 20.32 + 2.00 = **22.32 cm** (regression budget).

**H2 (breadth)**: BONN balloon, 100 frames stride=2, full pipeline.
- Accept if ATE RMSE ≤ **40.00 cm** (≥30% improvement over 57.35 baseline).
- Marginal-accept if 40.00 < ATE ≤ 52.00 cm.
- Reject if ATE > 57.35 + 3.00 = **60.35 cm**.

**H3 (safety)**: Replica room0, 100 frames stride=1, full pipeline.
- Accept if ATE RMSE ≤ **1.50 cm** (baseline ~1.2 cm; budget is non-regression).
- Reject if ATE > 2.00 cm (meaningful regression on static scene).

**Decision rule**: D1 is **accepted** only if H1 passes AND H2 does not
reject AND H3 does not reject. Marginal-accept on H1 requires H2 to
strictly pass. Any regression on H3 (static baseline) auto-rejects
regardless of H1/H2 because it indicates the change hurts the healthy
regime.

## Implementation

### Change 1: `dynlang_slam/core/losses.py::compute_losses`

Replace the soft alpha weighting of `rgb_mask` with a hard threshold
matching the depth mask. Introduce a configurable `reliability_thresh`
parameter (default 0.5) so both masks use the same rule.

```python
# Before:
alpha_rgb = alpha_2d.unsqueeze(-1).expand_as(pred_rgb)
rgb_mask = alpha_rgb

# After (hard-threshold reliability mask, matches depth):
reliable = (alpha_2d > reliability_thresh).float()
rgb_mask = reliable.unsqueeze(-1).expand_as(pred_rgb)
```

Depth path already has `alpha_2d > 0.5` hard gate — leave it, but plumb
the threshold through the same parameter for consistency.

### Change 2: `configs/default.yaml` loss weights

```yaml
loss:
  rgb_weight: 0.9     # was 0.5  (DG-SLAM λ₁)
  depth_weight: 0.1   # was 1.2  (DG-SLAM λ_depth)
  ssim_weight: 0.2    # unchanged (matches DG-SLAM λ₂)
  lang_weight: 0.1    # unchanged
```

### Change 3: `configs/default.yaml` new knob

```yaml
loss:
  reliability_thresh: 0.5  # hard gate for alpha-based reliability masking
```

## Why this is not the same failure pattern as C/B/B'/A1

| | C (silhouette) | B (depth warp) | B' (PIPs) | A1 (local BA) | D1 (this) |
|---|---|---|---|---|---|
| Introduces new upstream component? | no (hard gate) | yes (warp comp) | yes (CoTracker) | yes (BA optim) | **no** |
| Adds parameter DOF? | no | no | no | +24 DOF | **no** |
| Touches tracker loss formulation? | yes (hard cutoff) | no | no | no | **yes (soft→hard)** |
| Assumes upstream we don't have? | SplaTAM 200k init | DROID-VO init | dense GT tracks | ICP anchor + large KF graph | **no** |

D1 is a **targeted loss-formulation change** with no new parameters, no
upstream dependency, and no additional DOF. It does not share the
failure mode of the prior four rejections.

The closest analog is **C (silhouette gate)**, which also used a hard
alpha cutoff — but C applied it at `alpha > 0.95` **over the entire
rendered image**, which collapsed gradient in regions where no
Gaussians had been placed yet. D1 uses `alpha > 0.5` and only affects
the loss weighting of *already-produced gradients*, not whether pixels
contribute at all. Crucially, depth loss has been using exactly this
rule without issue for the entire project — we are harmonizing RGB to
match, not introducing a novel gate.

## Risks

1. **Loss scale shift**: Hard-thresholding RGB mask reduces the number
   of contributing pixels. Loss magnitude changes. Adam should adapt,
   but learning rate may need tuning. Smoke test watches for NaN /
   divergence.

2. **Loss weight flip may need H-regime adjustment**: If the new
   weights (depth_weight 0.1) leave depth too weak early in tracking
   (convex basin argument), tracker may fail to bootstrap. The tracker
   already has a time-varying weight schedule in `tracker.py` (lines
   161–170) that ramps RGB. This may interact. Smoke test watches
   early-frame ATE.

3. **Threshold sensitivity**: `reliability_thresh=0.5` is DG-SLAM-like
   but untested in our regime. If H1 fails, a secondary ablation at
   `thresh=0.3` (looser) is permitted before rejection, analogous to
   the PIPs `thresh_px` tuning permitted there.

## Budget

Small: no new modules, one function edited, config edit. Smoke test
~3 minutes, three benchmarks ~40 minutes total GPU time. Can be
executed and adjudicated in under an hour.

## Result tracking

- Smoke test log → `research/experiments/opacity-reliability/smoke_log.txt`
- H1 log → `research/experiments/opacity-reliability/h1_person_tracking.txt`
- H2 log → `research/experiments/opacity-reliability/h2_balloon.txt`
- H3 log → `research/experiments/opacity-reliability/h3_replica_room0.txt`
- Final verdict → `research/experiments/opacity-reliability/analysis.md`
