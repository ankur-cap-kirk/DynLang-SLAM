# BDGS-SLAM — Probabilistic 3D Gaussian Splatting for Robust SLAM in Dynamic Environments (2025)

Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12610981/

## Per-Gaussian Bayesian belief
```
β(s_t)  =  η · p(o_t | s_t) · β̄(s_t)                        (Bayes update)
p(o_t | s_t) = σ(w^T · f_t + b)                               (logistic lik.)
β̄(s_t = dyn)  =  e^{-α · Δt} · ( β(s_{t-1}) - 0.5 )  +  0.5   (temporal decay)
```
- `f_t`: concatenated semantic score (YOLOv5) + geometric features (position, cov, appearance).
- `α`: decay rate (value not disclosed).
- Evidence source: **still YOLOv5**. So BDGS is not class-agnostic either — just a smarter filter over YOLO.

## Tracker
- Photometric: `L = (1 - γ) · L1(I_r, I_gt) + γ · (1 - SSIM(I_r, I_gt))`.
- 20 tracker iters, 40 mapping iters.
- No explicit robust weighting, no failure detection.

## Drift control without loop closure
- No loop closure, no pose graph.
- Relies on: tightly-coupled tracking/mapping, multi-view Bayesian fusion across keyframes, EMA temporal smoothing.

## Ablation (Table 6, BONN averaged)
- Full: **1.82 cm ATE**
- Without Bayesian filter: **65.70 cm (36× worse)**
- Without multi-view: **32.22 cm (18× worse)**
- Without mapping: 2.85 cm

## Relevance to DynLang-SLAM
- Confirms the Bayesian belief framework is high-impact **when paired with good per-frame evidence**.
- But the evidence is still YOLO → doesn't solve our balloon gap on its own.
- Per-Gaussian belief already exists in our code; the gap is the *evidence update function*, which should include depth-warp residual and photometric re-rendering residual, not just YOLO scores.
