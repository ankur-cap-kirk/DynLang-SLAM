# Loss Weighting in 3DGS SLAM Tracking — Literature Survey

## Summary of Methods

### SplaTAM (CVPR 2024) — MOST RELEVANT
- **Tracking loss**: `L = sum_p [S(p)>0.99] (L1_depth + 0.5 * L1_color)`
- **Depth weight: 1.0, RGB weight: 0.5** (depth is 2x RGB)
- **Iterations**: 40 per frame (10 for fast variant)
- **Loss scheduling**: NONE — constant weights throughout
- **Key insight**: Depth gets FULL weight, RGB gets HALF. Silhouette masking (>0.99) restricts loss to well-mapped regions

### MonoGS (CVPR 2024 Highlight)
- Uses direct optimization against Gaussians
- Exact weights not found in search, but follows similar depth-heavy pattern
- No loss scheduling mentioned

### Gaussian-SLAM (2024)
- **Tracking loss**: `L = lambda_c * L1_color + (1-lambda_c) * L1_depth`
- **All weights set to 1.0** (equal depth and color)
- **Loss scheduling**: NONE

### RTG-SLAM (SIGGRAPH 2024)
- **Does NOT use loss-based tracking** — uses ICP (point-to-plane)
- Gaussian optimization: color_w=1, depth_w=1, reg_w=1000
- No loss scheduling

### GS-SLAM (CVPR 2024)
- Photometric loss weight: 0.8
- Uses coarse-to-fine tracking (sparse pixels first)
- LR scheduling: higher LR for first 5 iterations, then lower
- No loss WEIGHT scheduling

### CG-SLAM (2024)
- Uses depth uncertainty model to select valuable Gaussians
- No specific loss weight scheduling mentioned

## Key Pattern: NO PAPER USES LOSS WEIGHT ANNEALING

Every single 3DGS SLAM system uses **CONSTANT loss weights** during tracking.
The universal pattern is:
- Depth weight >= RGB weight (depth is the anchor)
- SplaTAM: depth=1.0, rgb=0.5 (depth is 2x)
- Gaussian-SLAM: depth=1.0, rgb=1.0 (equal)
- No system reduces depth weight during optimization

## What This Means for DynLang-SLAM

Our dynamic loss weighting (depth 2.0→0.5) VIOLATES the established pattern.
Reducing depth below 1.0 in late iterations removes the geometric anchor.

### Recommended Approach
Based on literature, the correct strategy is:
1. **Keep depth weight constant at 1.2** (our tuned baseline)
2. **Only modulate RGB/SSIM upward** in late iterations (additive boost, not substitution)
3. This way depth always provides the geometric anchor, RGB adds refinement on top

### Specific Schedule Proposal
- depth_scale(t) = 1.0 (constant, never changes)
- rgb_scale(t) = 0.5 + 0.5*t (0.5 → 1.0: starts at half, rises to full)
- ssim_scale(t) = 0.5 + 0.5*t (same as RGB)

This matches SplaTAM's philosophy (depth=1.0, rgb=0.5) in early iterations,
then gradually adds photometric refinement without ever reducing depth.

## Sources
- [SplaTAM](https://arxiv.org/abs/2312.02126) — CVPR 2024
- [MonoGS](https://arxiv.org/abs/2312.06741) — CVPR 2024 Highlight
- [Gaussian-SLAM](https://arxiv.org/abs/2312.10070) — 2024
- [RTG-SLAM](https://arxiv.org/abs/2404.19706) — SIGGRAPH 2024
- [GS-SLAM](https://arxiv.org/abs/2311.11700) — CVPR 2024
- [CG-SLAM](https://zju3dv.github.io/cg-slam/) — 2024
