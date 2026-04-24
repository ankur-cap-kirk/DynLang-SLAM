"""Full DynLang-SLAM test on BONN: Language + Dynamic Masking.

Runs the complete pipeline with BOTH features enabled:
  - YOLOv8 dynamic object detection + temporal filtering
  - CLIP + SAM language feature extraction + autoencoder
  - 3D text queries on the final Gaussian map

Compares three modes:
  1. Static (no dynamic, no language)
  2. Dynamic only (YOLO masking, no language)
  3. Full (dynamic + language)

Then runs 3D text queries to find objects in the scene.
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
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

# ---- Config ----
# SEQUENCE can be overridden via DYNLANG_SEQUENCE env var (just the BONN subdir name)
_SEQ_NAME = os.environ.get("DYNLANG_SEQUENCE", "rgbd_bonn_person_tracking")
SEQUENCE = os.path.join(PROJECT_ROOT, "data", "BONN", _SEQ_NAME)
N_FRAMES = int(os.environ.get("DYNLANG_N_FRAMES", "100"))
STRIDE = int(os.environ.get("DYNLANG_STRIDE", "2"))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "bonn_full_test")

# Text queries for 3D search after SLAM
TEXT_QUERIES = [
    "person", "human", "wall", "floor", "table", "chair",
    "door", "monitor", "screen", "shelf", "window",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"DynLang-SLAM Full Test: BONN {_SEQ_NAME}")
print(f"  {N_FRAMES} frames, stride={STRIDE}")
print(f"  Language: ON | Dynamic: ON")
print("=" * 60)


def run_slam(mode: str) -> tuple[dict, "SLAMPipeline | None", "GaussianMap | None"]:
    """Run SLAM in a given mode.

    Args:
        mode: "static", "dynamic", or "full"

    Returns:
        (results_dict, slam_pipeline, gaussian_map)
        slam and gaussian_map are only returned for "full" mode (for queries).
    """
    dynamic_on = mode in ("dynamic", "full")
    language_on = mode == "full"
    print(f"\n{'='*60}")
    print(f"--- Mode: {mode.upper()} (dynamic={'ON' if dynamic_on else 'OFF'}, language={'ON' if language_on else 'OFF'}) ---")
    print(f"{'='*60}")

    # Load config
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0

    # Dynamic settings
    cfg.dynamic.enabled = dynamic_on
    cfg.dynamic.yolo_model = "yolov8n-seg"  # lightweight

    # Language settings
    cfg.language.enabled = language_on
    if language_on:
        cfg.language.extract_every_n = 2          # extract every 2nd keyframe
        cfg.language.scales = ["whole"]            # single scale (8GB GPU friendly)
        cfg.language.autoencoder.warmup_frames = 30  # lower warmup for 100-frame test
        cfg.language.sam_checkpoint = os.path.join(
            PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"
        )

    # BONN intrinsics
    intrinsics = get_bonn_intrinsics()

    # Dataset
    dataset = TUMDataset(
        data_dir=SEQUENCE,
        height=480, width=640,
        depth_scale=5000.0,
        max_frames=N_FRAMES,
        stride=STRIDE,
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
    t_start = time.time()
    first_frame = dataset[0]
    slam.process_first_frame(gaussian_map, first_frame)

    ates = []
    dynamic_pcts = []
    lang_extractions = 0

    for i in range(1, len(dataset)):
        frame = dataset[i]
        info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
        ates.append(info["ate"])
        dynamic_pcts.append(info.get("dynamic_pct", 0.0))

        if "lang_extract_time" in info:
            lang_extractions += 1

        if i % 10 == 0 or i == len(dataset) - 1:
            parts = [f"Frame {i:3d}/{len(dataset)-1}",
                     f"ATE={info['ate']*100:.2f}cm",
                     f"G={info['total_gaussians']}"]
            if dynamic_on:
                parts.append(f"dyn={info.get('dynamic_pct', 0):.1f}%")
            if "lang_loss" in info:
                parts.append(f"lang_loss={info['lang_loss']:.4f}")
            if "lang_extract_time" in info:
                parts.append(f"lang={info['lang_extract_time']:.1f}s")
            if "contamination_cleaned" in info:
                parts.append(f"cleaned={info['contamination_cleaned']}")
            print(f"  {' | '.join(parts)}")

    total_time = time.time() - t_start
    gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    ate_rmse = slam.compute_ate_rmse(gt_poses) * 100  # cm

    # Language feature stats
    lang_stats = {}
    if language_on:
        lang_feats = gaussian_map.lang_feats.data
        nonzero = (lang_feats.abs().sum(dim=-1) > 0.01).sum().item()
        lang_stats = {
            "lang_feat_shape": list(lang_feats.shape),
            "lang_nonzero": nonzero,
            "lang_nonzero_pct": nonzero / lang_feats.shape[0] * 100,
            "lang_norm_min": lang_feats.norm(dim=-1).min().item(),
            "lang_norm_max": lang_feats.norm(dim=-1).max().item(),
            "lang_extractions": lang_extractions,
            "ae_frozen": slam._autoencoder.is_frozen if slam._autoencoder else False,
        }

    results = {
        "mode": mode,
        "ate_rmse_cm": ate_rmse,
        "total_gaussians": gaussian_map.num_gaussians,
        "total_time": total_time,
        "n_frames": len(dataset),
        "avg_dynamic_pct": np.mean(dynamic_pcts) if dynamic_pcts else 0,
        **lang_stats,
    }

    print(f"\n  {mode.upper()} Results:")
    print(f"    ATE RMSE:   {ate_rmse:.2f} cm")
    print(f"    Gaussians:  {gaussian_map.num_gaussians}")
    print(f"    Time:       {total_time:.1f}s ({total_time/len(dataset):.2f}s/frame)")
    if dynamic_on:
        print(f"    Avg dynamic: {results['avg_dynamic_pct']:.1f}%")
    if language_on and lang_stats:
        print(f"    Lang feats:  {lang_stats['lang_nonzero']}/{lang_stats['lang_feat_shape'][0]} "
              f"non-zero ({lang_stats['lang_nonzero_pct']:.1f}%)")
        print(f"    Lang norm:   [{lang_stats['lang_norm_min']:.4f}, {lang_stats['lang_norm_max']:.4f}]")
        print(f"    AE frozen:   {lang_stats['ae_frozen']}")
        print(f"    Extractions: {lang_stats['lang_extractions']}")

    # Only keep slam+map alive for "full" mode (for queries)
    if mode == "full":
        return results, slam, gaussian_map
    else:
        del gaussian_map, slam
        torch.cuda.empty_cache()
        return results, None, None


# ---- Run all 3 modes ----
results_static, _, _ = run_slam("static")
results_dynamic, _, _ = run_slam("dynamic")
results_full, slam_full, gmap_full = run_slam("full")

# ---- 3D Text Queries ----
print(f"\n{'='*60}")
print("3D Text Queries on Full Pipeline Map")
print(f"{'='*60}")

if slam_full._lang_initialized and slam_full._autoencoder is not None:
    for query in TEXT_QUERIES:
        try:
            # Relevancy scoring (contrastive, sharper)
            result = slam_full.query_3d(gmap_full, query, top_k=50, use_relevancy=True)
            scores = result["top_k_scores"]
            positions = result["top_k_positions"]
            center = positions.mean(dim=0)

            # Also get raw cosine
            result_raw = slam_full.query_3d(gmap_full, query, top_k=50, use_relevancy=False)
            raw_scores = result_raw["top_k_scores"]

            print(f"  '{query:10s}': relevancy=[{scores.min():.3f}, {scores.max():.3f}]  "
                  f"raw=[{raw_scores.min():.3f}, {raw_scores.max():.3f}]  "
                  f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        except Exception as e:
            print(f"  '{query:10s}': ERROR - {e}")
else:
    print("  Language pipeline not initialized.")
    print(f"    _lang_initialized = {slam_full._lang_initialized}")
    print(f"    _autoencoder = {slam_full._autoencoder}")
    if slam_full._autoencoder:
        print(f"    is_frozen = {slam_full._autoencoder.is_frozen}")
        print(f"    buffer_size = {slam_full._autoencoder.buffer_size}")

# ---- Final Comparison Table ----
print(f"\n{'='*60}")
print(f"Final Comparison: BONN {_SEQ_NAME}")
print(f"{'='*60}")
print(f"  {'Metric':<25s} {'Static':>10s} {'Dynamic':>10s} {'Full':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'ATE RMSE (cm)':<25s} {results_static['ate_rmse_cm']:>10.2f} {results_dynamic['ate_rmse_cm']:>10.2f} {results_full['ate_rmse_cm']:>10.2f}")
print(f"  {'Gaussians':<25s} {results_static['total_gaussians']:>10d} {results_dynamic['total_gaussians']:>10d} {results_full['total_gaussians']:>10d}")
print(f"  {'Time (s)':<25s} {results_static['total_time']:>10.1f} {results_dynamic['total_time']:>10.1f} {results_full['total_time']:>10.1f}")
print(f"  {'Time/frame (s)':<25s} {results_static['total_time']/results_static['n_frames']:>10.2f} {results_dynamic['total_time']/results_dynamic['n_frames']:>10.2f} {results_full['total_time']/results_full['n_frames']:>10.2f}")
print(f"  {'Avg dynamic (%)':<25s} {'N/A':>10s} {results_dynamic['avg_dynamic_pct']:>10.1f} {results_full['avg_dynamic_pct']:>10.1f}")

if "lang_nonzero_pct" in results_full:
    print(f"  {'Lang feat coverage (%)':<25s} {'N/A':>10s} {'N/A':>10s} {results_full['lang_nonzero_pct']:>10.1f}")

print(f"\n  SOTA reference: DG-SLAM = 4.73cm, BDGS-SLAM = 4.03cm")
print(f"{'='*60}")

# Cleanup
del gmap_full, slam_full
torch.cuda.empty_cache()
