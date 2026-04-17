# MonoGS — Gaussian Splatting SLAM (CVPR 2024)

Paper: https://arxiv.org/abs/2312.06741
Code: https://github.com/muskie82/MonoGS

## Tracking objective
- L1 photometric: `E_pho = || I(G, T_CW) - I_obs ||_1`
- L1 depth: `E_geo = || D(G, T_CW) - D_obs ||_1`
- RGB-D combined: `λ_pho · E_pho + (1 - λ_pho) · E_geo`, with `λ_pho = 0.9`
- No Huber. Pure L1. Affine brightness params also optimized for exposure changes.

## Tracking iteration schedule
- **100 iterations per frame** (both monocular and RGB-D).
- Early termination: pose update magnitude `< 1e-4`.

## Keyframe-window local BA (the key mechanism for us)
- **Joint optimization of keyframe poses + Gaussians during mapping.**
- Window size: **8 keyframes for TUM, 10 for Replica.**
- Objective: `Σ_{k ∈ W} E_pho^k  +  λ_iso · E_iso`
- Optimization variables: `{ T_CW^k ∈ SE(3) : k ∈ W } ∪ G`
- Keyframe selection (add to window): covisibility IoU `< 0.90` (TUM) / `0.95` (Replica), **OR** relative translation `>  0.08 × median_depth`.
- Removal from window: overlap coverage `< 0.3` with latest keyframe.

## Isotropic regularization
- `E_iso = Σ_i || s_i - s̄ · 1 ||_1` — penalizes scale deviation from mean scale, keeps Gaussians near-spherical.
- Weight: `λ_iso = 10`.
- Prevents view-aligned elongated Gaussians from creating artifacts.

## Failure detection / recovery
- **None.** Paper claims "large convergence basin" from direct 3DGS tracking, no explicit recovery.

## Relevance to DynLang-SLAM
- Direct blueprint for local BA: window of 8 KFs, joint pose+Gaussian optimization, pure PyTorch (no g2o).
- Our tracker currently **freezes KF poses after tracking** — this is the gap.
- Keyframe trigger by `0.08 × median_depth` is scale-invariant; ours is a fixed `0.05 m` — worth revisiting after local BA is in place.
