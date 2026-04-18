# Experiment: Depth-warp dynamic mask (intervention B)

## Motivation
After intervention C (silhouette gate) was rejected for regime mismatch, we
move to intervention B from `research/to_human/sota_gap_closing_methods.html`:
a **class-agnostic** motion mask that complements YOLO. This is the direct
cause of the balloon-sequence failure — YOLO has no "balloon" class in the
COCO-80 list we use, so today every balloon pixel is trusted as static and
gets baked into the map. A geometric (depth-warp) check does not need a class
label.

## What the change does
Port the depth-consistency mask from DG-SLAM (Xu et al., NeurIPS 2024, Eq. 5-7).

For each new frame j with depth D_j and velocity-init pose T_j:
1. For every recent frame i in a FIFO window of size N=4, back-project j's
   pixels to 3D, transform into i's camera frame via `T_ij = T_i^{-1} · T_j`,
   project to i's image, and bilinearly sample D_i at the projected location.
2. Flag pixels where `|D_i_sampled − z_pred| > 0.6 m` as dynamic **for that
   pair**.
3. Intersect (logical AND) across all pairs in the window → this gives a
   confident class-agnostic dynamic set, because static pixels that happen
   to disagree with one reference frame (occlusion, depth hole) are unlikely
   to disagree with ALL of them.
4. UNION with the YOLO semantic mask so we keep semantic knowledge where we
   have it. Output is a `1 = static, 0 = dynamic` float mask — the existing
   convention of `Tracker.track(mask=...)`.

Mapping, Bayesian belief, and contamination cleanup all consume the same
fused mask, so the fix is end-to-end.

## Citation
DG-SLAM (NeurIPS'24) Eq. 5-7. We implement the equivalent formulation and
keep their default threshold (0.6 m) and window (N=4). The buffer stores the
TRACKED pose after each frame, not the velocity-init guess, so errors don't
compound.

## Hypothesis
- **H1 (primary)**: On BONN `balloon` (baseline 57.35 cm full), a
  class-agnostic mask catches the balloon pixels that YOLO misses, and full
  ATE drops below 40 cm. This is where we expect the BIG win because it's a
  direct repair of a known failure.
- **H2 (safety)**: On BONN `person_tracking` (baseline 20.32 cm full), where
  YOLO's "person" class already covers the dynamic content, the warp mask is
  redundant. ATE should not regress by more than 2 cm. If it regresses more,
  the warp mask is poisoning static regions.
- **H0 (accept-criterion, pre-registered)**: H1 passes AND H2 does not
  regress more than the 2 cm safety budget.

## Code diff summary
- `dynlang_slam/dynamic/depth_warp.py` (NEW): `DepthWarpBuffer` FIFO +
  `_pairwise_dynamic_mask` + `fused_motion_mask` (intersect across window,
  union with YOLO).
- `dynlang_slam/dynamic/__init__.py`: exports.
- `dynlang_slam/slam/pipeline.py`:
  - `__init__` reads `cfg.dynamic.depth_warp.{enabled,window_size,threshold,min_window}`
    and allocates `self._warp_buffer = None` lazily.
  - `process_first_frame` seeds the buffer with frame 0's (depth, gt_pose).
  - `process_frame` computes the fused mask BEFORE tracking (using the
    velocity-init pose) and overwrites `dynamic_mask` with it.
  - After tracking, pushes (depth, est_pose) into the buffer for future
    frames. Tracked pose, not velocity-init, so error doesn't compound.
- `configs/default.yaml`: added `dynamic.depth_warp` block.

## Protocol
1. Run `python scripts/test_bonn_full.py` on `person_tracking` (100 frames,
   stride 2), log to `person_tracking_log.txt`.
2. Run same on `balloon`, log to `balloon_log.txt`.
3. Compare full-mode ATE to baselines:
   - person_tracking full: **20.32 cm**
   - balloon full: **57.35 cm**

## Accept criteria (pre-registered)
| Sequence         | Baseline | Accept if full ATE ≤ | Reason                   |
|------------------|---------:|---------------------:|--------------------------|
| balloon          |    57.35 |        **40.00**    | H1: big gain expected    |
| person_tracking  |    20.32 |        **22.32**    | H2: ≤ +2 cm drift budget |

If BOTH pass → accept, commit, proceed to intervention A (local BA).
If balloon passes but person_tracking regresses > 2 cm → tune threshold
(try 0.8 m, 1.0 m) before giving up, because the pairwise threshold is the
obvious knob.
If balloon fails → hypothesis wrong (something else is causing balloon
failure). Document and move to intervention A.

## Expected runtime
~5-10 min per sequence × 2 = ~20 min.

## What I am NOT doing in this experiment
- Not tuning `window_size` or `threshold`. Using DG-SLAM defaults (4, 0.6).
- Not touching mapper mask consumption (it already honors `dynamic_mask`).
- Not touching the Bayesian-belief update (already consumes `dynamic_mask`).

The question under test is **does the warp mask, plugged into the existing
infrastructure with paper-default hyperparameters, close the balloon gap**.
If the answer is "almost", then we tune. If the answer is "no", we don't.
