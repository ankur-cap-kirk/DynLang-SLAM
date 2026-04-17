# Analysis: Loss-Weight Tuning

## Result: Hypothesis NOT supported. Reverted to baseline A.

## Summary (BONN person_tracking, 100 frames, Full mode is primary)

| Config | depth_w | rgb_w | huber_depth_delta | Static | Dynamic | Full |
|--------|---------|-------|-------------------|--------|---------|------|
| A-rebaseline | 1.2 | 0.5 | 0.10 | 28.36 | 21.85 | **20.32** |
| E (low-depth)   | 0.6 | 0.5 | 0.10 | 29.04 | 25.09 | 24.23 |
| F (wide-Huber)  | 1.2 | 0.5 | 0.20 | 22.85 | 24.66 | 32.55 |
| G (combined)    | 0.8 | 0.7 | 0.15 | 31.54 | 21.10 | 25.32 |

All three variants regressed vs A in full mode (+4 to +12 cm).

## Key observations

1. **Full mode responded opposite to static mode.**
   F improved static by -5.5 cm but worsened full by +12 cm. The language
   pipeline is sensitive to loss balance — SAM-guided feature supervision
   relies on accurate geometry, which depth weight/Huber shifts disrupt.

2. **Map-growth starves with low depth fidelity.**
   Gaussian count stuck at ~18-20k in E/F (baseline reaches 33-37k).
   `_expand_map` uses depth silhouette for new-Gaussian placement, so weakening
   depth signal starves map growth. Tracker then has nothing to track against
   in late frames where camera explores new regions.

3. **Dynamic mode is stable (21-25 cm) across all configs.**
   The dynamic-mask pipeline is robust to loss weighting.

4. **Variance is ~3 cm between repeated A runs,** but configs E/F/G regressed
   by 4-12 cm — clearly outside the noise floor. Signal is real.

## Revised hypothesis

Late-frame drift is NOT caused by Kinect noise dominating the loss. The
real bottleneck is:

**Map-growth during exploration.** When the camera visits new regions at
frame 70+, `_expand_map` can't add Gaussians fast enough. New Gaussians
take multiple mapping iterations to stabilize. Meanwhile the tracker tries
to align to a stale/sparse map and lands in bad basins, propagating drift.

## Next experiment candidates

1. **Increase mapping iterations** in exploration regime (currently 60).
   Hypothesis: more mapping time → better new-Gaussian convergence →
   tracker has stable map to follow.

2. **Lower `new_gaussian_thresh`** (currently 0.5).
   Hypothesis: add Gaussians more aggressively in low-silhouette regions,
   compensating for Kinect depth gaps.

3. **Raise densification frequency** (`densify_interval: 10 -> 5`).
   Hypothesis: more frequent adaptive density control keeps map responsive.

## Decision

- Revert config to A (current commit: c30196c-parent baseline).
- Close this experiment directory.
- Open next experiment: "map-growth-tuning" targeting exploration regime.
