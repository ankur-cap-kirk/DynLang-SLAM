# Analysis: Silhouette gate (intervention C) — REJECTED

## Result

| Mode     | Baseline (cm) | Silhouette gate @ 0.95 (cm) | Δ    |
|----------|--------------:|----------------------------:|-----:|
| static   |         28.36 |                     **215.05** | +186.7 |
| dynamic  |         21.85 |                      **61.39** |  +39.5 |
| full     |         20.32 |                      **75.30** |  +55.0 |

Pre-registered accept threshold was `full ≤ 19.82 cm`. We got 75.30 cm.
**Reject. Revert.**

## Per-frame ATE trajectory (full mode)

| Frame | Baseline | Gate @ 0.95 |
|------:|---------:|------------:|
|    10 |    ~3 cm |       8.74  |
|    20 |   ~7 cm  |      34.18  |
|    30 |   ~9 cm  |      69.32  |
|    60 |   ~15 cm |      94.27  |
|    99 |   ~20 cm |     102.87  |

Drift starts earlier and grows faster. This is not a late-drift event — it's
systemic gradient starvation from frame ~10 onward.

## Diagnosis

The hard gate at alpha > 0.95 is too aggressive for our map regime:

1. In the first ~30 frames, the map is sparse (16–22k Gaussians). Most pixels
   have rendered alpha well below 0.95.
2. When the gate excludes the majority of pixels, the loss collapses toward
   zero, gradients become sparse and noisy, and the pose optimizer can't
   correct meaningful pose error.
3. Since we initialize each new frame from constant velocity, any
   un-corrected error propagates forward, and the gate prevents recovery on
   subsequent frames where alpha would naturally grow.

## Why literature supported this but we can't use it as-is

- SplaTAM (CVPR'24) uses `S(p) > 0.99` and succeeds because:
  - Their first frame is initialized with 200k+ Gaussians from a dense depth
    point cloud (vs our ~16k).
  - They run 40 tracker iters AND do aggressive per-frame mapping, so alpha
    coverage is near-saturated within a few frames.
  - Their loss weights (`depth=1.0, rgb=0.5`) emphasize depth, which is more
    tolerant of sparse coverage than RGB photometric.
- MonoGS (CVPR'24) **does not use a hard gate at all**. It uses soft alpha
  weighting (multiply residual by continuous alpha).
- Our existing implementation already uses soft alpha weighting. Switching to
  a hard gate is thus not an upgrade for our regime — it's a regime mismatch.

## Lessons

1. Even "principled" interventions from literature assume a specific operating
   regime (map density, iteration budget, initialization). Porting them
   requires checking whether the regime matches ours.
2. The SOFT alpha weighting already in our code is load-bearing for tracker
   stability. Don't remove it.
3. For a 2-week project, we should skip any intervention that requires
   re-engineering the map-density regime. Move to interventions that are
   strictly additive (depth-warp mask, local BA).

## Action
- Revert `compute_losses`, `Tracker`, `pipeline.py`, `default.yaml` on this
  branch.
- Branch stays unmerged.
- Proceed directly to intervention B: depth-warp dynamic mask (DG-SLAM).

## Files
- `protocol.md` — pre-registered hypothesis and accept criteria.
- `person_tracking_log.txt` — full run log.
