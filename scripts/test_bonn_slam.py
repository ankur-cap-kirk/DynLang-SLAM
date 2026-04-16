"""End-to-end SLAM test on BONN RGB-D Dynamic (person_tracking).

Runs the full DynLang-SLAM pipeline with dynamic masking enabled:
  1. YOLOv8 detects the walking person
  2. Temporal filter confirms dynamic regions
  3. Tracker ignores dynamic pixels
  4. Mapper excludes dynamic regions from Gaussians
  5. Contaminated Gaussians are cleaned up

Compares ATE with and without dynamic masking.
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
SEQUENCE = os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_person_tracking")
N_FRAMES = 100  # use 100 frames for a meaningful test
STRIDE = 2      # every other frame (sequence is 30fps, so ~15fps)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "bonn_slam_test")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"BONN SLAM Test: person_tracking ({N_FRAMES} frames, stride={STRIDE})")
print("=" * 60)


def run_slam(dynamic_enabled: bool) -> dict:
    """Run SLAM pipeline and return metrics."""
    tag = "dynamic" if dynamic_enabled else "static"
    print(f"\n--- Running SLAM ({tag} mode) ---")

    # Load config and override for BONN
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0

    # Use lighter YOLO for speed
    cfg.dynamic.enabled = dynamic_enabled
    cfg.dynamic.yolo_model = "yolov8n-seg"

    # Disable language features for this test (focus on dynamic masking)
    cfg.language.enabled = False

    # BONN intrinsics
    intrinsics = get_bonn_intrinsics()

    # Dataset
    dataset = TUMDataset(
        data_dir=SEQUENCE,
        height=480,
        width=640,
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
    for i in range(1, len(dataset)):
        frame = dataset[i]
        info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
        ates.append(info["ate"])

        dyn_pct = info.get("dynamic_pct", 0.0)
        dynamic_pcts.append(dyn_pct)

        if i % 10 == 0 or i == len(dataset) - 1:
            dyn_str = f" | dyn={dyn_pct:.1f}%" if dynamic_enabled else ""
            gauss_str = f" | G={info['total_gaussians']}"
            clean_str = ""
            if "contamination_cleaned" in info:
                clean_str = f" | cleaned={info['contamination_cleaned']}"
            print(f"  Frame {i:3d}/{len(dataset)-1} | "
                  f"ATE={info['ate']*100:.2f}cm{gauss_str}{dyn_str}{clean_str}")

    total_time = time.time() - t_start
    gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    ate_rmse = slam.compute_ate_rmse(gt_poses) * 100  # cm

    results = {
        "tag": tag,
        "ate_rmse_cm": ate_rmse,
        "total_gaussians": gaussian_map.num_gaussians,
        "total_time": total_time,
        "n_frames": len(dataset),
        "avg_dynamic_pct": np.mean(dynamic_pcts) if dynamic_pcts else 0,
        "per_frame_ate": ates,
    }

    print(f"\n  {tag.upper()} Results:")
    print(f"    ATE RMSE: {ate_rmse:.2f} cm")
    print(f"    Gaussians: {gaussian_map.num_gaussians}")
    print(f"    Time: {total_time:.1f}s ({total_time/len(dataset):.2f}s/frame)")
    if dynamic_enabled:
        print(f"    Avg dynamic area: {results['avg_dynamic_pct']:.1f}%")

    # Cleanup GPU
    del gaussian_map, slam
    torch.cuda.empty_cache()

    return results


# ---- Run both modes ----
results_static = run_slam(dynamic_enabled=False)
results_dynamic = run_slam(dynamic_enabled=True)

# ---- Comparison ----
print(f"\n{'='*60}")
print("BONN SLAM Comparison: Static vs Dynamic Masking")
print(f"{'='*60}")
print(f"  Sequence: person_tracking ({results_static['n_frames']} frames)")
print(f"")
print(f"  {'Metric':<25s} {'No Masking':>12s} {'With Masking':>12s}")
print(f"  {'-'*25} {'-'*12} {'-'*12}")
print(f"  {'ATE RMSE (cm)':<25s} {results_static['ate_rmse_cm']:>12.2f} {results_dynamic['ate_rmse_cm']:>12.2f}")
print(f"  {'Gaussians':<25s} {results_static['total_gaussians']:>12d} {results_dynamic['total_gaussians']:>12d}")
print(f"  {'Time (s)':<25s} {results_static['total_time']:>12.1f} {results_dynamic['total_time']:>12.1f}")
print(f"  {'Avg dynamic area (%)':<25s} {'N/A':>12s} {results_dynamic['avg_dynamic_pct']:>12.1f}")

improvement = results_static['ate_rmse_cm'] - results_dynamic['ate_rmse_cm']
if improvement > 0:
    print(f"\n  Dynamic masking IMPROVED ATE by {improvement:.2f} cm ({improvement/results_static['ate_rmse_cm']*100:.1f}%)")
else:
    print(f"\n  Dynamic masking changed ATE by {improvement:.2f} cm")

print(f"{'='*60}")
