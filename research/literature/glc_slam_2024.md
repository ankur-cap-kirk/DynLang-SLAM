# GLC-SLAM — Gaussian Splatting SLAM with Efficient Loop Closure (2024)

Paper: https://arxiv.org/abs/2409.10982

## Loop detection
- **NetVLAD** descriptor per keyframe (pre-trained).
- Global + local keyframe databases.
- **Cosine similarity** between descriptors.
- Two levels:
  - Global: new submap triggers global query, accept best match above dynamic threshold.
  - Local: during mapping, accept most similar KF above predefined threshold.
- False-positive rejection: **geometric check** using Jaccard index (frame overlap) of observed Gaussians between candidates.

## Pose graph
- Nodes: keyframe SE(3) poses.
- Edges:
  - Sequential (consecutive KF relative poses).
  - Loop (loop-closing KF pairs).
- Optimization: Levenberg-Marquardt on `v* = argmin ½ · Σ e_i^T · Λ_i^{-1} · e_i`.

## Map update after LC
- Keyframe-centric: each Gaussian tied to the KF that spawned it.
- When KF pose changes from `T → T'`:
  ```
  μ' = T' · T^{-1} · μ
  R' = R_{T'} · R_T^{-1} · R
  ```
- Followed by mapping-loss refinement with densification/pruning DISABLED.

## Ablation (Replica room0, Table VII)
- Full: 0.20 cm ATE
- Without LC: 0.29 cm (+45% relative, +0.09 cm absolute)
- → Loop closure gives **modest absolute improvement** on short trajectories.

## Relevance to DynLang-SLAM
- BONN person_tracking is **short + mostly linear** → limited loop opportunities.
- Our failure mode is a **discrete drift event at frame ~65**, not slow drift across loops.
- Conclusion: loop closure has high implementation cost (NetVLAD, pose graph with LM solver, map deformation) for probably <2 cm ATE gain on our benchmarks.
- **Skip for 2-week timeline. Prefer local BA + failure recovery.**
