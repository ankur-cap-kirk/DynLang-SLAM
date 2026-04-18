# Experiment: CoTracker-based dynamic mask (intervention B')

## Motivation
Course Lecture 27 (EEE 515, Warping and Tracking) explicitly introduces
PIPs (Harley et al., ECCV 2022) as the modern occlusion-robust alternative
to optical flow and classical features. The lecture's framing — "point
tracking handles occlusion; flow doesn't" — is exactly our class-agnostic
masking problem. We use **CoTracker2** (Karaev et al., 2023), the direct
descendant of PIPs from the same Meta research group. CoTracker2 is
available via `torch.hub.load("facebookresearch/co-tracker", "cotracker2")`,
is pretrained (frozen, 195 MB checkpoint), and runs in ~770 ms for an 8-frame
window × 400 query points on our GPU.

## Relation to prior rejections

Intervention B (depth-warp, DG-SLAM) failed because its residual depended
on the velocity-init pose estimate: 5 cm translation × 3 m depth → ~15 cm
depth residual, already a meaningful fraction of the 0.6 m threshold, so the
signal was dominated by pose error. The intersection-across-window step
then erased true positives along with the noise.

CoTracker avoids this failure mode:

1. **Observations are pose-independent.** CoTracker produces 2D trajectories
   from RGB alone. The OBSERVATION of where a pixel moved is not
   contaminated by our tracker's pose estimate.
2. **Residual is in pixel space.** 1-2 px of residual from pose error is
   small compared to dynamic objects' 10-50 px per-frame displacements,
   giving a cleaner SNR than the depth-metric residual of intervention B.
3. **We use TRACKED poses, not velocity-init.** The mask is computed AFTER
   tracking the current frame, so all poses used for reprojection are the
   tracker's refined output.
4. **No intersection-AND step.** We aggregate the residual across the
   window (mean / max) and threshold, rather than requiring unanimous
   agreement across pairs. Median-correction removes the pose-error bias.

## What the change does

**Algorithm:**

1. Maintain a sliding buffer of the last T frames (RGB + depth + tracked
   pose). T = 8 (CoTracker's native window).
2. After tracking each frame, if the buffer has ≥ T frames, run CoTracker
   with a 20×20 grid of query points seeded on frame 0 of the window (the
   OLDEST, which has a known tracked pose) and predict forward.
3. For each query point, compute the **predicted-static** 2D position on
   each frame in the window: back-project via K⁻¹ + depth at the query
   seed, transform via `T_i⁻¹ · T_0`, reproject with K.
4. Compute the per-frame residual `|observed - predicted|` in pixels.
   Median-correct to remove camera-egomotion bias.
5. Aggregate: a point is dynamic if its max-frame residual > `thresh_px`
   (default 5 px).
6. Upsample sparse 20×20 dynamic classification to dense (H, W) mask via
   nearest-neighbor + small dilation.
7. UNION with YOLO semantic mask. Same output convention as before
   (1 = static, 0 = dynamic).

The mask is computed AFTER tracking, so it does NOT affect the current
frame's tracker. It IS used for mapping, Bayesian-belief update, and
contamination cleanup. This is a deliberate design choice: we want a
clean first-principles test of whether a better class-agnostic mask
helps the MAP, not a coupled test with the tracker.

## Hypothesis

- **H1 (primary)**: On BONN `balloon` (baseline full 57.35 cm),
  CoTracker catches the balloon pixels that YOLO misses. Full ATE drops
  below **40 cm** (same threshold as B).
- **H2 (safety)**: On BONN `person_tracking` (baseline full 20.32 cm),
  CoTracker is redundant with YOLO's person class. ATE does not regress
  by more than 2 cm (≤ 22.32 cm).
- **H0 (diagnostic)**: Before claiming H1/H2, we require that CoTracker
  actually adds detections beyond YOLO. Specifically: `dyn%_{PIPs+YOLO}` -
  `dyn%_{YOLO}` averaged across frames must exceed 0.5 pp. If the PIPs
  signal adds nothing, we reject on grounds of ineffectiveness before
  spending ATE-run budget.

## Code diff summary

- `dynlang_slam/dynamic/pips_mask.py` (NEW): `PointTrackBuffer` + inference
  wrapper around CoTracker2 + predicted-vs-observed residual + sparse-to-
  dense upsample.
- `dynlang_slam/dynamic/__init__.py`: exports.
- `dynlang_slam/slam/pipeline.py`: init CoTracker lazily (hub download +
  GPU load) on first keyframe where we have T buffered frames. After
  tracking each frame, compute and UNION the PIPs mask with the existing
  `dynamic_mask`. Log per-frame additive contribution (`pips_added_pct`).
- `configs/default.yaml`: `dynamic.pips` block.

## Protocol

1. Run `scripts/smoke_pips_mask.py` standalone first: verify PIPs adds
   dynamic detections on 2-3 BONN balloon frames (diagnostic pre-check
   for H0).
2. If H0 passes → run `scripts/test_bonn_full.py` on person_tracking,
   log to `person_tracking_log.txt`.
3. Run on balloon, log to `balloon_log.txt`.

## Accept criteria (pre-registered)

| Sequence         | Baseline | Accept if full ATE ≤ | Reason                   |
|------------------|---------:|---------------------:|--------------------------|
| balloon          |    57.35 |        **40.00**     | H1: class-agnostic win   |
| person_tracking  |    20.32 |        **22.32**     | H2: ≤ +2 cm budget       |

**Diagnostic H0**: `avg(pips_added_pct) ≥ 0.5 pp` across the run.
If H0 fails, the PIPs signal isn't informative — reject before
interpreting ATE.

If BOTH H1 and H2 pass → accept, commit, write up.
If H0 fails → reject; document that CoTracker in this regime adds no
class-agnostic signal.
If H1 fails but H0 and H2 pass → tune threshold / grid density before
giving up.

## Runtime budget

- CoTracker inference: ~770 ms per call on 8-frame × 400-point window
- Called once per frame after tracking
- Expected overhead: ~77 s / 100 frames → full run ~280 s (4.7 min) per
  sequence per mode, ~30 min total for 3 modes × 2 sequences.

## What I am explicitly NOT doing

- NOT training CoTracker. Frozen pretrained weights only.
- NOT using the PIPs mask in the current frame's tracker (deliberate: test
  map effect first).
- NOT tuning threshold / grid density before the first run. DG-SLAM
  defaults analog: 5 px (≈ 1% of image width), 20×20 grid (400 points).

The question under test is: **does a pretrained course-covered point
tracker, plugged into the mapping/belief pipeline with sensible defaults,
close the balloon gap without regressing person_tracking?**
