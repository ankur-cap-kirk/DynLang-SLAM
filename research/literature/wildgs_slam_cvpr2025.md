# WildGS-SLAM — Monocular Gaussian Splatting SLAM in Dynamic Environments (CVPR 2025)

Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_WildGS-SLAM_Monocular_Gaussian_Splatting_SLAM_in_Dynamic_Environments_CVPR_2025_paper.pdf

## Core idea: class-agnostic uncertainty via DINOv2
- Pre-trained DINOv2 features per pixel → shallow MLP → per-pixel uncertainty.
- **No YOLO, no class labels, no segmentation.** Foundation feature model handles open-world cases.
- High-uncertainty pixels (dynamic regions, occlusions, unmapped) get downweighted in both tracking and dense BA.

## MLP details
- Trained online as frames arrive.
- Optimized independently of Gaussian map.

## Loss weighting
- Uncertainty reweights photometric loss in both tracking and mapping.
- Exact formula not in paper excerpt; likely `w_p = 1 / (σ_p^2 + ε)` style precision weighting.

## Cost vs benefit
- Extra cost: DINOv2 forward pass per frame (~300 MB model, ~100ms on good GPU).
- Extra module: shallow MLP (~few layers).
- BONN numbers not in the excerpt read.

## Relevance to DynLang-SLAM
- Cleanest class-agnostic story in the literature.
- HIGH ceiling but HIGH cost: adds a heavy feature backbone and an online-trained MLP.
- For a 2-week timeline: **skip**. The depth-warp mask (DG-SLAM) gets most of the class-agnostic benefit at a fraction of the complexity.
