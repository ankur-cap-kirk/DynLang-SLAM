"""Test language-enabled SLAM with 3D text queries.

Runs SLAM for 50 frames with language extraction, then queries the map
for objects like "chair", "table", etc.
"""

import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
from pathlib import Path

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

N_FRAMES = 50
SCENE = "room0"
QUERIES = ["chair", "table", "floor", "wall", "plant", "sofa", "window", "lamp"]

print("=" * 60)
print(f"Language SLAM Test: {SCENE} ({N_FRAMES} frames)")
print("=" * 60)

# Load config
cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
cfg.dataset.scene = SCENE
cfg.dataset.max_frames = N_FRAMES
cfg.language.extract_every_n = 2  # extract every other keyframe
cfg.language.autoencoder.warmup_frames = 15  # lower warmup for short test

# Dataset
dataset_path = Path(PROJECT_ROOT) / cfg.dataset.path / SCENE
dataset = ReplicaDataset(
    data_dir=str(dataset_path),
    height=cfg.dataset.image_height,
    width=cfg.dataset.image_width,
    depth_scale=cfg.dataset.depth_scale,
    max_frames=N_FRAMES,
)
intrinsics = get_replica_intrinsics(
    fx=cfg.camera.fx, fy=cfg.camera.fy,
    cx=cfg.camera.cx, cy=cfg.camera.cy,
    height=cfg.dataset.image_height,
    width=cfg.dataset.image_width,
)

# Init
device = "cuda"
gaussian_map = GaussianMap(
    sh_degree=cfg.gaussians.sh_degree,
    lang_feat_dim=cfg.gaussians.lang_feat_dim,
    init_opacity=cfg.gaussians.init_opacity,
    device=device,
)
slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)

# Run SLAM
print("\n--- Running SLAM ---")
t0 = time.time()
first_frame = dataset[0]
slam.process_first_frame(gaussian_map, first_frame)

for i in range(1, len(dataset)):
    frame = dataset[i]
    info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
    if i % 10 == 0:
        lang_info = f" | lang_loss={info.get('lang_loss', 0):.4f}" if 'lang_loss' in info else ""
        print(f"  Frame {i}/{len(dataset)-1} | ATE={info['ate']*100:.2f}cm{lang_info}")

total_time = time.time() - t0
ate = slam.compute_ate_rmse([f["pose"].to(device) for f in [dataset[i] for i in range(len(dataset))]]) * 100
print(f"\nSLAM done: ATE={ate:.2f}cm | {gaussian_map.num_gaussians} Gaussians | {total_time:.1f}s")

# Check lang_feats
lang_feats = gaussian_map.lang_feats.data
print(f"\nLanguage features: shape={lang_feats.shape}")
print(f"  Non-zero: {(lang_feats.abs().sum(dim=-1) > 0.01).sum().item()} / {lang_feats.shape[0]}")
print(f"  Norm range: [{lang_feats.norm(dim=-1).min():.4f}, {lang_feats.norm(dim=-1).max():.4f}]")

# 3D Text Queries (with relevancy scoring)
print("\n--- 3D Text Queries (relevancy scoring) ---")
if slam._lang_initialized and slam._autoencoder is not None:
    for query in QUERIES:
        try:
            # Relevancy scoring (contrastive, sharper)
            result = slam.query_3d(gaussian_map, query, top_k=50, use_relevancy=True)
            scores = result["top_k_scores"]
            positions = result["top_k_positions"]
            center = positions.mean(dim=0)

            # Also get raw cosine for comparison
            result_raw = slam.query_3d(gaussian_map, query, top_k=50, use_relevancy=False)
            raw_scores = result_raw["top_k_scores"]

            print(f"  '{query}': relevancy=[{scores.min():.3f}, {scores.max():.3f}] "
                  f"raw=[{raw_scores.min():.3f}, {raw_scores.max():.3f}] "
                  f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        except Exception as e:
            print(f"  '{query}': ERROR - {e}")
else:
    print("  Language pipeline not initialized (too few frames?)")

print(f"\n{'='*60}")
print("Test complete!")
