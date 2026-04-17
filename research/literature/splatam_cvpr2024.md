# SplaTAM — Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM (CVPR 2024)

Paper: https://spla-tam.github.io/

## Tracking loss
```
L_t = Σ_p  [ S(p) > 0.99 ]  ·  ( L1(D(p)) + 0.5 · L1(C(p)) )
```
- **Silhouette gating**: only pixels where rendered alpha silhouette `S(p) > 0.99` contribute to the loss.
- Silhouette `S(p) = Σ_i f_i(p) · Π_{j<i} (1 - f_j(p))` (alpha composite).
- Depth weighted 1.0, color weighted 0.5.
- No explicit Huber/M-estimator; the silhouette mask serves as outlier rejection.

## Why silhouette matters
- Rejects unmapped / poorly-fit regions before they contaminate the pose gradient.
- **Key insight for DynLang-SLAM**: when our tracker plateaus on hard frames, a lot of that plateau is from rendered-but-not-well-mapped pixels. Silhouette gate would suppress those contributions.

## Iterations
- 40 tracker iters per frame (SplaTAM).
- 10 iters for SplaTAM-S (fast variant).
- **No adaptive schedule.**

## Keyframes
- Saved every Nth frame.
- Mapping: `k` keyframes optimized jointly — current frame + most recent KF + `k-2` KFs with highest frustum overlap.
- Overlap = project current depth pointcloud into candidate KF frustum, count inliers.

## Pose initialization
- **Constant velocity**: `E_{t+1} = E_t + (E_t - E_{t-1})`.
- First frame: identity.

## Depth-only vs RGB-only ablation
- Depth-only → "completely fails" (no x-y information).
- RGB-only → succeeds but `> 5x` error vs both.
- Validates using both terms with current (0.9 RGB / 0.1 depth) or (1.0 depth / 0.5 RGB) weighting.

## Relevance to DynLang-SLAM
- Silhouette gating is **~20 LoC** drop-in for our tracker: in our photometric residual, multiply by `(rendered_alpha > 0.95).float()`.
- Cheap, orthogonal to every other change we'd make.
