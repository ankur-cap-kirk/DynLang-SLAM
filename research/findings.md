# Language Pipeline Findings — DynLang-SLAM

## Current Understanding

The language pipeline for 3DGS follows a well-established architecture pioneered by LERF (NeRF-based) and refined by LangSplat and LEGaussians (3DGS-based). The core pipeline is:

```
Image → SAM masks (3 scales) → Crop per mask → CLIP encode → 768/512-dim features
    → Autoencoder compress → 3/16-dim latent → Store in Gaussians
    → Render latent features via splatting → Decode → Cosine similarity with text query
```

## Key Methods Compared

### LangSplat (CVPR 2024 Highlight) — Primary Reference
- **SAM masks**: 32x32 grid of point prompts → 3 semantic levels (subpart, part, whole)
- **CLIP model**: OpenCLIP ViT-B/16 (512-dim output)
- **Autoencoder**: MLP-based, 512-dim → 3-dim latent (they chose 3 for RGB visualization)
- **AE loss**: L1 + cosine distance loss on reconstructed CLIP features
- **Per-Gaussian storage**: 3 language embeddings (one per scale: subpart, part, whole)
- **Rendering**: Same tile-based rasterizer as color, splatting language features
- **Querying**: Relevancy score with canonical phrase contrastive anchors ("object", "things", "stuff", "texture")
- **Training**: 25 min on RTX 3090, ~4GB memory, 30K iterations
- **Speed**: 199x faster than LERF at 1440x1080

### LEGaussians (CVPR 2024) — Quantization Approach
- **Key difference**: Uses quantization instead of autoencoder for compression
- **Features**: Multi-view CLIP + DINO features, quantized to discrete codebook
- **Advantage**: No per-scene autoencoder training needed
- **Disadvantage**: Quantization can lose fine-grained semantic nuance

### LERF (ICCV 2023) — NeRF Baseline
- **Approach**: Volume-renders CLIP embeddings in NeRF, multi-scale supervision
- **Disadvantage**: 199x slower than LangSplat, requires DINO features
- **Historical importance**: First language-embedded 3D representation

### OpenScene (CVPR 2023) — Point Cloud Approach
- **Approach**: Back-project 3D points into images, aggregate CLIP features
- **Trains**: 3D sparse convnet to predict CLIP features from geometry alone
- **Relevance**: Multi-view fusion strategy applicable to our Gaussian pipeline

## Architecture Decisions for DynLang-SLAM

### What We Should Implement (LangSplat-style)

**1. CLIP Feature Extraction**
- Use OpenCLIP ViT-L/14 (768-dim) — already in our config
- Extract every N keyframes (config: `extract_every_n: 5`)
- For each SAM mask: crop image × mask → CLIP encode → 768-dim feature

**2. SAM Multi-Scale Masks**
- Generate SAM masks with automatic mode (grid of point prompts)
- Three semantic levels: subpart, part, whole
- Each pixel gets 3 CLIP features (one per level)
- Remove redundant masks by IoU/stability scores

**3. Autoencoder (768 → 16-dim)**
- Our config uses 16-dim (vs LangSplat's 3-dim) — more capacity for semantics
- Architecture: MLP encoder (768 → 256 → 16) + MLP decoder (16 → 256 → 768)
- Loss: L1 + cosine distance on reconstructed CLIP features
- Training: Online during SLAM warmup (first 100 frames per config)
- After warmup: freeze autoencoder, continue encoding new features

**4. Fusion into Gaussians**
- Each Gaussian gets 16-dim `lang_feats` (field already exists, all zeros)
- During mapping: render language features via splatting, supervise against 2D CLIP maps
- Language features optimized jointly with geometry during mapping iterations

**5. Open-Vocabulary Querying**
- Text query → CLIP text encoder → 768-dim → autoencoder encoder → 16-dim
- Cosine similarity against all Gaussians' `lang_feats`
- Threshold → highlight matching Gaussians
- Use canonical phrases for contrastive relevancy scoring

### Key Design Choices

| Decision | LangSplat | Our Choice | Reason |
|----------|-----------|------------|--------|
| Latent dim | 3 | 16 | More semantic capacity; LangSplat used 3 for RGB viz |
| CLIP model | ViT-B/16 (512d) | ViT-L/14 (768d) | Better features, we have the config |
| AE training | Offline, per-scene | Online, during SLAM | SLAM is incremental, can't pre-train |
| Scales | 3 (subpart/part/whole) | 3 | Follow LangSplat |
| SAM model | ViT-H | ViT-L | Per our config, good speed/quality tradeoff |

### Critical Challenge: Online Autoencoder Training

LangSplat trains the autoencoder offline on all images first. In SLAM, we must train it **online** as frames arrive:

1. **Warmup phase** (frames 0-100): Collect CLIP features from keyframes, train autoencoder
2. **Encoding phase** (frames 100+): Freeze autoencoder, encode new features in real-time
3. **Risk**: Early features may not represent the full scene → autoencoder may not generalize
4. **Mitigation**: Continue collecting features and periodically fine-tune autoencoder

### Implementation Order

1. **CLIP feature extraction** on keyframes (standalone, testable)
2. **SAM mask generation** at 3 scales (standalone, testable)
3. **Autoencoder** training pipeline (offline first, then online)
4. **Mapper integration** — supervise lang_feats during mapping
5. **3D text query** — cosine similarity + visualization

## Open Questions

1. Is 16-dim enough to preserve fine-grained CLIP semantics? (LangSplat uses 3-dim and works, so 16 should be more than enough)
2. How to handle autoencoder warmup in SLAM where you can't see the full scene upfront?
3. Should we use all 3 SAM scales or simplify to 1 for SLAM speed?
4. How much does language feature rendering add to per-frame time?

---

# Dynamic Masking Improvement — BONN person_tracking

## Problem Statement

Our current binary YOLO + temporal filter dynamic masking **makes ATE worse**: 42.32cm (dynamic) vs 35.03cm (static-only) on BONN person_tracking. SOTA methods achieve **4.03–4.73cm** on the same sequence. The gap is ~10x.

## SOTA Benchmarks on BONN person_tracking

| Method | ATE RMSE (cm) | Year | Key Approach |
|--------|---------------|------|--------------|
| DG-SLAM | 4.73 | NeurIPS 2024 | Depth warp mask + coarse-to-fine tracking |
| BDGS-SLAM | 4.03 | 2025 | Bayesian per-Gaussian belief + soft opacity |
| DGS-SLAM | ~5-6 | 2024 | Photometric residual histogram |
| **Ours (static)** | **35.03** | — | No dynamic handling |
| **Ours (YOLO mask)** | **42.32** | — | Binary YOLO mask, hurts tracking |

## Root Cause Analysis

### Why binary masking hurts tracking:

1. **Signal loss**: Removing 11% of pixels reduces photometric + depth signal for pose optimization. The tracker has fewer pixels to work with → noisier gradients → worse poses.

2. **No geometric verification**: YOLO masks are purely semantic. They mask out furniture (misclassified) or miss partially-visible people. The masks don't correspond to actual motion.

3. **The tracker already tolerates small dynamics**: Our coarse-to-fine tracking with L1 loss is somewhat robust to outliers. Aggressively removing pixels can hurt more than the dynamics do.

4. **No contamination cleanup**: Even with masking, dynamic pixels leak into Gaussians during the frames before detection starts. Those contaminated Gaussians persist forever.

## 6 Concrete Fixes (Ranked by Impact)

### FIX 1: Depth Warp Mask (Highest Impact — from DG-SLAM)
**What**: Instead of semantic-only masks, verify dynamics *geometrically* by projecting depth between keyframes and checking for residuals.

**How** (~50 lines in `tracker.py`):
```python
def compute_depth_warp_mask(depth_cur, depth_prev, pose_cur, pose_prev, K, threshold=0.1):
    """Project current depth into previous frame, compare residuals."""
    H, W = depth_cur.shape
    # Create pixel grid
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    # Unproject current frame pixels to 3D
    z = depth_cur
    x = (u - K[0,2]) * z / K[0,0]
    y = (v - K[1,2]) * z / K[1,1]
    pts_3d = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)  # (H,W,4)
    # Transform to previous frame
    T_rel = pose_prev.inverse() @ pose_cur
    pts_prev = (T_rel @ pts_3d.reshape(-1,4).T).T.reshape(H,W,4)
    # Project to previous frame pixels
    u_prev = (pts_prev[...,0] / pts_prev[...,2]) * K[0,0] + K[0,2]
    v_prev = (pts_prev[...,1] / pts_prev[...,2]) * K[1,1] + K[1,2]
    # Sample previous depth at projected locations
    depth_sampled = F.grid_sample(depth_prev.unsqueeze(0).unsqueeze(0), ...)
    # Compare: large residual = dynamic
    residual = (pts_prev[...,2] - depth_sampled.squeeze()).abs()
    dynamic_mask = residual > threshold * z  # relative threshold
    return dynamic_mask
```

**Why it works**: Moving objects physically change depth. A walking person's depth changes between frames while static walls don't. This catches *actual motion*, not just semantic categories.

**Expected impact**: This is the single biggest improvement. DG-SLAM attributes most of their performance to this mask (Eq. 5-7 in their paper). Combined with semantic mask: `final_mask = depth_warp_mask | semantic_mask`.

### FIX 2: Soft Per-Pixel Weighting (High Impact — from DGS-SLAM)
**What**: Instead of binary mask (0 or 1), assign continuous weights to each pixel based on photometric residual.

**How** (~30 lines in `losses.py`):
```python
def compute_robust_weights(rendered_rgb, gt_rgb, percentile=90):
    """Photometric residual histogram → soft weights."""
    residual = (rendered_rgb - gt_rgb).abs().mean(dim=0)  # (H, W)
    threshold = torch.quantile(residual.flatten(), percentile / 100.0)
    # Soft sigmoid around threshold (not hard cutoff)
    weights = 1.0 - torch.sigmoid(10.0 * (residual - threshold) / threshold)
    return weights  # (H, W), high weight = likely static
```

Then in `compute_losses()`:
```python
robust_weights = compute_robust_weights(rendered["rgb"], gt_rgb)
if mask is not None:
    robust_weights = robust_weights * mask  # combine with any existing mask
loss = (robust_weights * pixel_loss).mean()
```

**Why it works**: Dynamic pixels have high photometric residual (the person moved). Instead of removing them entirely, we down-weight them smoothly. This preserves tracking signal from partially-occluded areas while reducing dynamic influence.

### FIX 3: Coarse-to-Fine with Better Initialization (High Impact)
**What**: Use constant-velocity motion model for initial pose guess instead of identity/previous-frame.

**How** (~15 lines in `tracker.py`):
```python
# In track(), before optimization:
if len(self.pose_history) >= 2:
    # Constant velocity model
    T_prev = self.pose_history[-1]
    T_prev2 = self.pose_history[-2]
    delta = T_prev @ T_prev2.inverse()
    initial_pose = delta @ T_prev  # extrapolate
else:
    initial_pose = self.pose_history[-1] if self.pose_history else torch.eye(4)
```

**Why it works**: Camera motion is smooth. Starting optimization from a velocity-extrapolated pose means fewer iterations needed and less chance of getting stuck in local minima. DG-SLAM uses DROID-SLAM for this, but constant-velocity is much simpler and often sufficient.

### FIX 4: Accumulated Opacity Mask for Tracking (Medium Impact — from DG-SLAM)
**What**: Only track against pixels where Gaussians have been well-observed (high accumulated opacity from splatting).

**How** (~5 lines in `tracker.py`):
```python
# After rendering, before loss computation:
opacity_map = rendered["opacity"]  # (H, W) from Gaussian splatting
reliable_mask = opacity_map > 0.95  # only well-reconstructed pixels
if mask is not None:
    mask = mask & reliable_mask
else:
    mask = reliable_mask
```

**Why it works**: Low-opacity regions are poorly reconstructed (often because they contain dynamics, or were only seen briefly). Tracking against them adds noise. DG-SLAM's Eq. 9 shows this is a key component.

### FIX 5: Bayesian Per-Gaussian Dynamic Belief (Medium Impact — from BDGS-SLAM)
**What**: Each Gaussian maintains a dynamic probability β ∈ [0,1] that decays over time. Used to suppress dynamic Gaussians' opacity during rendering.

**How** (~40 lines in `gaussians.py`):
```python
# New buffer in GaussianMap:
self.dynamic_belief = torch.zeros(N, device=device)  # β per Gaussian

def update_dynamic_belief(self, viewmat, K, W, H, dynamic_mask, decay=0.95):
    """Project Gaussians to 2D, update belief based on mask overlap."""
    # Project Gaussian means to pixel coords
    means_2d = project_to_2d(self.means, viewmat, K, W, H)
    # Check which land in dynamic region
    in_dynamic = sample_mask(dynamic_mask, means_2d)  # bool
    # Bayesian update with temporal decay
    self.dynamic_belief[in_dynamic] = 1.0 - decay * (1.0 - self.dynamic_belief[in_dynamic])
    self.dynamic_belief[~in_dynamic] *= decay  # decay toward static

def get_effective_opacity(self):
    """Suppress dynamic Gaussians during rendering."""
    return self.opacities * (1.0 - self.dynamic_belief)
```

**Why it works**: Instead of binary prune/keep decisions, this softly suppresses Gaussians that are repeatedly observed in dynamic regions. Gaussians that were briefly occluded by a walking person recover via decay. BDGS-SLAM achieves 4.03cm with this approach.

### FIX 6: Depth-Verified Semantic Masking (Easy Win)
**What**: Only trust YOLO detections where the detected object's depth is inconsistent with the static map (i.e., don't mask stationary furniture that YOLO thinks is "dynamic").

**How** (~10 lines in `pipeline.py`):
```python
# After YOLO detection, before using mask:
if len(detections) > 0 and prev_depth is not None:
    for det in detections:
        mask_region = det["mask"]
        # Check if depth in masked region differs from previous frame
        depth_diff = (cur_depth[mask_region] - prev_depth[mask_region]).abs().mean()
        if depth_diff < 0.05:  # less than 5cm average depth change
            det["mask"][:] = False  # not actually moving, remove mask
```

**Why it works**: YOLO might detect a "person" who is standing still (not moving). Or it might detect a sofa as a dynamic object. Depth verification confirms actual motion.

## Recommended Implementation Order

1. **FIX 6** (depth-verified semantic mask) — 10 min, easy win, prevents false positives
2. **FIX 3** (constant-velocity init) — 15 min, improves all tracking, not just dynamic
3. **FIX 2** (soft weights) — 30 min, replaces binary masking entirely
4. **FIX 1** (depth warp mask) — 1-2 hours, biggest single improvement
5. **FIX 4** (opacity mask) — 15 min, complements depth warp
6. **FIX 5** (Bayesian belief) — 1-2 hours, full cleanup solution

**Expected combined result**: Implementing FIX 1-4 should bring ATE from ~35cm down to **8-15cm** range. Adding FIX 5-6 could push toward **5-8cm**, closer to SOTA.

## Sources

- [DG-SLAM: Robust Dynamic Gaussian Splatting SLAM](https://arxiv.org/abs/2411.08373) — NeurIPS 2024
- [BDGS-SLAM: Bayesian Dynamic 3D Gaussian Splatting SLAM](https://arxiv.org/html/2503.11921) — 2025
- [DGS-SLAM: Dynamic 3D Gaussian Splatting SLAM](https://github.com/junzhejiang16/DGS-SLAM) — 2024
- [LangSplat: 3D Language Gaussian Splatting](https://arxiv.org/abs/2312.16084) — CVPR 2024 Highlight
- [LEGaussians: Language Embedded 3D Gaussians](https://arxiv.org/abs/2311.18482) — CVPR 2024
- [LERF: Language Embedded Radiance Fields](https://arxiv.org/abs/2303.09553) — ICCV 2023
- [OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/abs/2211.15654) — CVPR 2023
- [LangSplat GitHub](https://github.com/minghanqin/LangSplat)
- [LEGaussians GitHub](https://github.com/buaavrcg/LEGaussians)
- [Online Language Splatting](https://arxiv.org/html/2503.09447)
- [Gen-LangSplat](https://arxiv.org/abs/2510.22930) — Generalized autoencoder approach
