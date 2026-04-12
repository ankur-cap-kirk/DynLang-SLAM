# DynLang-SLAM: Complete Project Explainer
## For Team Onboarding Meeting

---

## What Are We Building?

We are building **DynLang-SLAM** — a system that does three things simultaneously:

1. **SLAM** (Simultaneous Localization and Mapping) — A camera moves through a room. The system figures out WHERE the camera is (localization) and WHAT the room looks like (mapping) at the same time.

2. **Language Queries** — After building the map, you can type "where is the chair?" and the system highlights it in 3D. Like Ctrl+F but for the real world.

3. **Dynamic Object Handling** — If a person walks through the scene, the system ignores them instead of crashing. Current systems completely break when anything moves.

**No existing system does all three together. That's our contribution.**

---

## Why Does This Matter?

Imagine a robot navigating your house:
- It needs to know where it is → **SLAM**
- It needs to understand "go to the kitchen table" → **Language**
- It needs to not crash when your dog runs by → **Dynamic handling**

Current systems can do ONE of these. We're combining all three.

---

## The Core Technology: 3D Gaussian Splatting (3DGS)

### What is a Gaussian?

Think of a 3D Gaussian as a **fuzzy colored blob** in 3D space. Each blob has:
- **Position** (x, y, z) — where it is
- **Size & Shape** (scale + rotation) — how big and in what orientation (can be stretched like an ellipsoid)
- **Color** (RGB) — what color it is
- **Opacity** — how transparent it is (0 = invisible, 1 = solid)

### How does a scene get built?

We represent the entire room as **thousands of these blobs** (typically 50,000–100,000). From a distance, they look like a photorealistic image. Think of it like pointillism painting — thousands of dots that form a picture.

### Why not NeRF?

NeRF (Neural Radiance Fields) was the previous approach. It uses a neural network to represent the scene. Problems:
- **Slow**: Rendering one image takes seconds (we need milliseconds)
- **Implicit**: The scene is locked inside a neural network — you can't easily add/remove objects
- **Not differentiable enough**: Hard to optimize camera poses through it

3DGS is **100x faster** (real-time at 100+ FPS) and **explicit** (each Gaussian is a tangible thing you can modify).

### How does rendering work?

```
3D Gaussians → Project onto 2D image plane → Sort by depth → Alpha-composite front to back → Final image
```

This is called **splatting** — you "splat" each 3D blob onto the camera image. The gsplat library does this on GPU in <1ms.

**Key insight**: This rendering is **differentiable** — you can compute gradients through it. This means you can ask "if I move the camera slightly left, how does the rendered image change?" and use that to optimize the camera position.

---

## How SLAM Works (Module 1)

### The Tracking-Mapping Loop

For every new camera frame:

```
Frame N arrives (RGB + Depth image)
    │
    ▼
TRACKING (estimate WHERE the camera is)
    │  - Start from previous frame's pose as initial guess
    │  - Render the Gaussian map from that pose
    │  - Compare rendered image vs. actual image
    │  - Compute loss (how different are they?)
    │  - Backpropagate through the renderer
    │  - Update the camera pose to reduce the loss
    │  - Repeat for 50 iterations
    │
    ▼
Is this a keyframe? (every 5th frame)
    │
    YES ──────────────────────────┐
    │                             │
    ▼                             ▼
MAPPING (improve the map)     Skip mapping
    │  - Optimize Gaussian params
    │    (positions, colors, sizes)
    │  - Add new Gaussians for
    │    unseen regions
    │  - Prune dead Gaussians
    │    (opacity ≈ 0)
    │
    ▼
Next frame
```

### Tracking in Detail

**Input**: Current Gaussian map + new RGB-D image
**Output**: Camera pose (4×4 matrix: rotation + translation)

The camera pose is parameterized as:
- **Quaternion** (4 numbers) — represents rotation (w, x, y, z)
- **Translation** (3 numbers) — represents position (x, y, z)

These 7 numbers are optimized using Adam optimizer with a combined loss:

```
Total Loss = 0.5 × L1_RGB + 1.0 × L1_Depth + 0.2 × (1 - SSIM)
```

- **L1 RGB**: Pixel-by-pixel color difference
- **L1 Depth**: Pixel-by-pixel depth difference (this is crucial — without depth, tracking is unreliable)
- **SSIM**: Structural similarity (captures texture/edge similarity, not just pixel values)

**LR Scheduling**: We use cosine annealing — the learning rate starts high (0.002) for fast convergence, then smoothly decays to 30% of initial value to avoid overshooting the optimal pose.

### Mapping in Detail

**Input**: Keyframe images + estimated poses
**Output**: Updated Gaussian map

Two phases:
1. **Optimize existing Gaussians** — adjust positions, colors, sizes to better match observations
2. **Add new Gaussians** — for regions the camera sees for the first time

**How we detect new regions (Silhouette-based expansion)**:
```
Render alpha map (opacity) at 8x downscale
Find pixels where alpha < 0.5 (not covered by existing Gaussians)
AND depth > 0 (valid depth measurement)
→ Unproject these pixels to 3D using depth + camera intrinsics
→ Create new Gaussians at those 3D positions
```

### What We've Achieved So Far

| Scene | ATE (cm) | Quality |
|-------|----------|---------|
| room0 | 0.72 | Sub-centimeter! |
| office0 | 0.96 | Sub-centimeter! |
| office4 | 1.09 | Sub-centimeter! |
| office2 | 0.59 | Best scene |
| office3 | 2.01 | Great |
| room1 | 10.34 | Needs work (drift on fast motion) |
| room2 | 18.82 | Needs work |

SplaTAM (our baseline) gets ~0.3-0.5cm on easy scenes. We're competitive on most scenes.

---

## How Language Embedding Works (Module 2 — Week 2)

### The Goal

Attach a "meaning vector" to each Gaussian so we can search the 3D map with text.

### The Pipeline

```
RGB Frame
    │
    ▼
SAM (Segment Anything Model)
    │  Generates 3 levels of masks:
    │  - Large (whole objects: "couch")
    │  - Medium (parts: "couch cushion")
    │  - Small (details: "couch button")
    │
    ▼
CLIP (Contrastive Language-Image Pre-training)
    │  For each mask region, extract a 768-dim feature vector
    │  This vector captures the "meaning" of what's in that region
    │  CLIP was trained on 400M image-text pairs from the internet
    │
    ▼
Autoencoder (768D → 16D)
    │  768 dimensions per Gaussian × 100,000 Gaussians = too much memory
    │  We train a small neural network to compress 768D → 16D
    │  This is scene-specific (trained on the first ~100 frames)
    │
    ▼
Per-Gaussian Language Feature (16D)
    │  Each Gaussian now has: position + color + size + opacity + 16D language vector
    │
    ▼
Query Time:
    User types: "find the chair"
    → CLIP encodes text → 768D → autoencoder decoder → 16D
    → Cosine similarity with every Gaussian's 16D feature
    → Gaussians with high similarity light up in the 3D map
```

### Why 16 dimensions?

- CLIP outputs 768D per region
- 768 × 100,000 Gaussians × 4 bytes = **307 MB** just for language features
- 16 × 100,000 Gaussians × 4 bytes = **6.4 MB** — manageable
- The autoencoder is trained to preserve semantic similarity in the compressed space

### Why SAM masks?

Without SAM, CLIP features are extracted per-pixel, which is noisy and doesn't capture object-level semantics. SAM provides clean object boundaries at multiple scales.

---

## How Dynamic Masking Works (Module 3 — Week 3)

### The Problem

If a person walks in front of the camera:
1. The tracking loss is computed on ALL pixels, including the person
2. The person moves between frames → the loss says "the camera must have moved a lot!"
3. The tracker estimates a WRONG pose → **drift begins**
4. Wrong pose → wrong Gaussians added → **map corruption**
5. Corrupted map → worse tracking → **cascading failure**

One moving object can destroy the entire trajectory.

### The Solution

```
RGB Frame
    │
    ▼
YOLOv8-Seg (Instance Segmentation)
    │  Detects: person, car, dog, cat, bicycle, etc. (COCO classes)
    │  Outputs: per-pixel binary mask (1 = dynamic, 0 = static)
    │
    ▼
Temporal Consistency Filter
    │  Don't trust a single detection — require object to be detected
    │  in 2 out of 3 consecutive frames (sliding window)
    │  This suppresses false positives (e.g., a painting of a person)
    │
    ▼
Alpha-Masked Loss
    │  Tracking loss is computed ONLY on static pixels:
    │  loss = L1(rendered[static_mask], gt[static_mask])
    │  Dynamic pixels contribute zero gradient
    │
    ▼
Gaussian Contamination Cleanup (every 20 frames)
    │  Some Gaussians may have been created ON the dynamic object
    │  before it was first detected. Find and remove them:
    │  - Project all Gaussians into recent frames
    │  - If a Gaussian lands inside the dynamic mask in 3+ frames
    │    → it's contaminated → prune it
```

### Known Risks (Be Honest About These)

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Over-masking** | YOLO detects a painting of a person → masks valid static geometry → less signal for tracking | Temporal filter (require 2/3 detections) |
| **Under-masking** | Only torso detected, not arms → unmasked arm pixels corrupt tracking | Mask dilation (expand mask by 5px) |
| **Closed vocabulary** | YOLO only knows COCO classes. A robot arm, spinning fan, or rolling ball won't be detected | This is a fundamental limitation. Could explore open-vocab detectors like Grounding-DINO |
| **Contamination delay** | Gaussians created on a person BEFORE first detection → corrupt map | Cleanup pass prunes contaminated Gaussians retroactively |
| **VRAM** | YOLOv8 (~500MB) + gsplat + CLIP all on one GPU (12GB) | Run YOLOv8 at lower resolution, share GPU memory carefully |
| **Replica has no dynamic objects** | Can't test dynamic masking on Replica | Use Bonn RGB-D Dynamic or TUM RGB-D datasets |

---

## Technical Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Rendering | gsplat (CUDA) | Differentiable Gaussian splatting, <1ms per frame |
| Deep Learning | PyTorch 2.10 + CUDA 12.8 | Optimization, autograd |
| Language | OpenCLIP ViT-L/14 | 768D image feature extraction |
| Segmentation | SAM (vit_l) | Multi-scale object masks |
| Detection | YOLOv8x-Seg | Dynamic object instance segmentation |
| GPU | RTX 5070 Laptop (12GB VRAM) | All computation |
| Dataset | Replica (static), Bonn/TUM (dynamic) | Evaluation |

---

## Project Structure

```
DynLang-SLAM/
├── configs/
│   └── default.yaml           # All hyperparameters
├── dynlang_slam/
│   ├── core/
│   │   ├── gaussians.py       # GaussianMap class (stores all Gaussians)
│   │   ├── renderer.py        # gsplat rendering (RGB + depth in single pass)
│   │   └── losses.py          # L1, SSIM, depth loss functions
│   ├── slam/
│   │   ├── tracker.py         # Camera pose optimization
│   │   ├── mapper.py          # Gaussian map optimization + expansion
│   │   └── pipeline.py        # Orchestrates tracking → mapping loop
│   ├── language/              # [WEEK 2] CLIP + SAM + autoencoder
│   └── dynamic/               # [WEEK 3] YOLOv8 + masking + cleanup
├── data/Replica/              # Dataset (8 scenes)
├── evaluate.py                # Run metrics across all scenes
├── visualize.py               # Generate comparison images
└── run.py                     # Main entry point
```

---

## How to Run

```powershell
# From project root
.\run.bat run.py                           # Run SLAM on default scene (room0)
.\run.bat evaluate.py --scenes room0 room1 # Evaluate on specific scenes
.\run.bat visualize.py                     # Generate visual comparisons
```

The `run.bat` script sets up MSVC compiler + CUDA environment automatically.

---

## What Each Teammate Could Own

| Role | Scope | Files |
|------|-------|-------|
| **SLAM & Tracking** | Improve tracking accuracy on hard scenes, LR scheduling, densification | tracker.py, mapper.py, gaussians.py |
| **Language Pipeline** | CLIP integration, autoencoder training, SAM masks, query interface | language/ (new module) |
| **Dynamic Masking** | YOLOv8 integration, temporal filtering, contamination cleanup | dynamic/ (new module) |
| **Evaluation & Paper** | Metrics, ablations, visualization, final report writing | evaluate.py, visualize.py, paper/ |

---

## Key Papers to Read

1. **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023) — The foundational paper. Read Section 3 (representation) and Section 5 (densification).
2. **SplaTAM** (Keetha et al., CVPR 2024) — Our SLAM architecture is based on this. Read the tracking-mapping loop.
3. **LangSplat** (Qin et al., CVPR 2024) — Our language pipeline follows this. Read the autoencoder and SAM mask sections.
4. **DGS-SLAM** (Deng et al., 2024) — Inspiration for dynamic masking.
5. **MonoGS** (Matsuki et al., CVPR 2024) — Regularization techniques we can borrow.

---

## Glossary

| Term | Meaning |
|------|---------|
| **ATE RMSE** | Absolute Trajectory Error (Root Mean Square). Measures tracking accuracy in cm. Lower = better. |
| **PSNR** | Peak Signal-to-Noise Ratio. Measures rendering quality in dB. Higher = better. >30 dB is good. |
| **SSIM** | Structural Similarity Index. 0 to 1. >0.9 is good. |
| **Depth L1** | Average depth error in meters. Lower = better. |
| **Keyframe** | A frame selected for mapping (every 5th frame). Non-keyframes only do tracking. |
| **Pose** | Camera position + orientation. Represented as a 4×4 matrix (rotation + translation). |
| **c2w** | Camera-to-world transform. Converts camera coordinates to world coordinates. |
| **w2c / viewmat** | World-to-camera transform. Inverse of c2w. gsplat uses this. |
| **Quaternion** | A 4-number representation of rotation (w, x, y, z). Avoids gimbal lock. |
| **Alpha compositing** | Blending transparent layers front-to-back: C = T × α × color. T = transmittance (how much light passes through). |
| **Silhouette** | The alpha (opacity) map rendered from the Gaussian map. Low alpha = region not covered by Gaussians. |
| **Densification** | Adding more Gaussians where needed (splitting big ones, cloning small ones). |
| **Pruning** | Removing useless Gaussians (opacity ≈ 0 or too large). |
| **CLIP** | OpenAI model trained on 400M image-text pairs. Maps images and text to the same 768D space. |
| **SAM** | Meta's Segment Anything Model. Segments any object given a point/box prompt. |
| **COCO** | Common Objects in Context. 80 object categories used by YOLO. |
