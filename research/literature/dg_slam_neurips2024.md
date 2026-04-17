# DG-SLAM — Robust Dynamic Gaussian Splatting SLAM with Hybrid Pose Optimization (NeurIPS 2024)

Paper: https://arxiv.org/abs/2411.08373
Code: https://github.com/fudan-zvg/DG-SLAM

## Depth warp mask (class-agnostic, the key mechanism for us on balloon)

### Reprojection (Eq. 5)
For pixel `p_i` in frame `i` with depth `D_i(p_i)`, project into frame `j`:
```
p_{i→j} = K · T_ji · (K^{-1} · D_i(p_i) · [p_i; 1])
```
where `K` is intrinsics and `T_ji` is the relative transform `T_j · T_i^{-1}`.

### Dynamic decision (Eq. 6)
```
if  D_j(p_{i→j}) - D_i(p_i)  <  e_th:
    mark pixel as dynamic (inconsistent depth across frames)
```
- Window size: **N = 4** previous frames.
- Depth threshold: **e_th = 0.6** (units unspecified; likely meters in world scale after depth scaling).

### Mask fusion (Eq. 7)
```
M_j  =  ( M_{j,i}^{wd} ∩ M_{j,i-1}^{wd} ∩ ... ∩ M_{j,i-N}^{wd} )  ∪  M_j^{sg}
```
- Intersection of depth-warp masks across N previous frames gives static consistency.
- Union with semantic segmentation mask `M^{sg}` → final motion mask.
- Rationale: warp mask compensates for objects missed by the semantic model.

## Pose optimization (hybrid, coarse-to-fine)
- Coarse: DROID-VO dense bundle adjustment for initial pose (Eq. 8).
- Fine: Gaussian splatting render-based pose refinement with accumulated opacity masking (Eq. 9-10).
- Final tracking loss: `λ_1 · L_color + λ_2 · L_depth`, with `λ_1 = 0.9`, `λ_2 = 0.1`.
- **No resolution pyramid.**

## BONN ATE (Table 3)
- balloon: **3.7 cm**
- person_tracking: **4.5 cm**
- move_box2: **3.5 cm**

## Ablation (Table 6)
- Full model: 5.51 cm (averaged)
- Without depth warp mask: 6.40 cm (+16%)
- Without semantic mask: **15.27 cm (+177%)**
- → Semantic is primary, depth warp is a complementary +16% refinement.

## DROID-VO dependency
- Paper does not ablate without DROID-VO. The depth warp idea is independent of the tracker choice, but the **end-to-end DG-SLAM numbers bake in DROID-VO quality**, making them hard to attribute.

## Relevance to DynLang-SLAM
- **Depth warp mask is directly portable**: requires only current depth, previous depth, relative pose, intrinsics.
- Fills the balloon gap: class-agnostic, will fire on the balloon because its depth violates static-scene consistency.
- Honest expectation: +16% on already-masked sequences (person_tracking), bigger gain on sequences where YOLO misses (balloon).
