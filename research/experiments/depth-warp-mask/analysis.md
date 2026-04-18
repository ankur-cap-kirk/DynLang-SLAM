# Analysis: Depth-warp dynamic mask (intervention B) — REJECTED

## Result

### person_tracking (safety check, H2)

| Mode    | Baseline (cm) | Depth-warp (cm) | Δ     |
|---------|--------------:|----------------:|------:|
| static  |         28.36 |           27.94 |  -0.4 |
| dynamic |         21.85 |           28.05 | +6.2  |
| full    |         20.32 |       **25.24** | +4.9  |

Pre-registered H2 accept: full ≤ 22.32 cm (+2 cm safety budget). **Got 25.24. Fail by 2.9 cm.**

### balloon (primary hypothesis, H1)

| Mode    | Baseline (cm) | Depth-warp (cm) | Δ     |
|---------|--------------:|----------------:|------:|
| static  |         ~58   |           50.01 |  ≈-8  |
| dynamic |         ~55   |           56.52 | +1.5  |
| full    |         57.35 |       **60.62** | +3.3  |

Pre-registered H1 accept: full ≤ 40.00 cm. **Got 60.62. Fail by 20.6 cm.**

Both hypotheses rejected. Reverting.

## Diagnosis

### 1. The warp mask is not adding class-agnostic detections

`dyn=` percentages printed per frame are effectively **identical** to the
YOLO-only baseline on both sequences (person_tracking 2.5%, balloon 3.0%).
If the warp were flagging dynamic pixels that YOLO missed (the whole point of
intervention B), we would expect `dyn%` to increase.

Three possible causes:

**a) Velocity-init pose error dominates the depth residual.**
The warp mask is computed BEFORE tracking, using the constant-velocity pose
guess as the current-frame pose. Every pixel's reprojected depth residual is
polluted by the pose error. With a 5 cm translation error and typical scene
depths of 3-5 m, that by itself can exceed our 0.6 m threshold almost
everywhere — and the intersection across 4 frames then collapses to zero
(each frame has different directions of error, so the AND wipes them out).

**b) The intersection is too strict for our depth data.**
BONN depth has significant noise and holes from the Kinect v2. Per-pair
dynamic probability for truly-static pixels is well above zero because of
sensor noise. Intersecting across 4 frames demands agreement by all pairs,
which legit-static regions easily fail when one of the 4 frames has a depth
glitch nearby. This flips what should be "confident dynamic" into "zero
confident dynamic" because the AND wipes out pixels.

**c) The intersection over a velocity-init window is sensitive to the pose
used in each frame of the window.**
DG-SLAM stores the tracked pose for each reference frame (we do the same),
but the CURRENT frame's pose is velocity-init — not yet tracked. So the
window is asymmetric: 4 known-good reference poses vs. 1 noisy current pose.
Any bias in the current pose reappears in every pair's residual, and the
intersection doesn't filter it out.

### 2. person_tracking late-drift amplified, not suppressed

Looking at per-frame ATE:

| Frame | Baseline full | Warp full |
|------:|--------------:|----------:|
|    20 |          ~7   |     25.80 |
|    40 |          ~12  |     18.97 |
|    60 |          ~15  |     15.80 |
|    80 |          ~18  |     42.66 |
|    99 |          ~20  |     35.37 |

The discrete drift event still happens at frame ~70, and now starts EARLIER
(frame 20 already at 25 cm vs baseline ~7). The warp mask is introducing
mask noise in the early frames before the intersection has enough samples,
and is not helping during the drift event.

### 3. balloon is worse in "static" mode than baseline (50 vs 58)

Possible noise in the baseline measurement; the 57.35 came from a different
run with different keyframe selection RNG. The real comparison is full vs
full, which regresses +3.3 cm.

## Why literature supported this but we can't use it as-is

DG-SLAM's recipe includes coarse DROID-VO + fine GS refinement. Their warp
mask receives a much better current-frame pose estimate (from DROID-VO)
than our constant-velocity guess. Our pose error is the dominant signal in
the residual — so what DG-SLAM calls "motion evidence" is for us "pose
error evidence", and we'd be masking the pixels WHERE WE NEED TRACKING LOSS
MOST.

This is the same kind of regime mismatch that killed intervention C — a
literature method assumed an upstream component (coarse pose or dense init)
that we don't have.

## Lessons

1. The warp mask's residual signal-to-noise requires a pose good enough that
   pose error is much smaller than 0.6 m reprojection threshold. For us,
   5 cm translation at 3 m depth ≈ 15 cm depth residual — already a
   meaningful fraction of the threshold.
2. The intersection step is a hard AND. On noisy depth, that tends to erase
   true positives rather than filter false positives.
3. We should have designed the experiment with a DIAGNOSTIC first (log
   per-pair `dyn%` and intersection `dyn%` for a few frames), not a direct
   metric run. We'd have seen that the warp mask adds nothing BEFORE
   spending 20 min on two full ATE runs.
4. For a 2-week project: geometric class-agnostic masking that depends on
   having a better upstream pose estimate is not a plug-in upgrade for us.
   Skip and pursue methods that don't depend on upstream pose quality.

## Action
- Revert pipeline.py, __init__.py, configs/default.yaml on this branch.
- KEEP depth_warp.py as an unused module (it's correct code; future work
  that first lands a better tracker front-end could reactivate it).
- Branch `feat/depth-warp-mask` stays unmerged.
- Preserve the failed-experiment record by merging this analysis branch to
  main with `--no-ff` (same discipline as intervention C).
- Proceed to intervention A: MonoGS-style keyframe-window local BA.

## Files
- `protocol.md` — pre-registered hypothesis and accept criteria.
- `person_tracking_log.txt` — full run log for person_tracking.
- `balloon_log.txt` — full run log for balloon.
