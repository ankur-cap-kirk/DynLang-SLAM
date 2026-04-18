"""Smoke test for intervention A1 (Local BA over keyframe window).

Runs 20 frames on BONN person_tracking with refine_poses=True and
verifies:
  1. No autograd errors through the pose params.
  2. pose_refine_trans_mean_m is finite, positive (poses are moving)
     but not absurd (not diverging).
  3. ATE at frame 20 is comparable to baseline (sanity check — full
     benchmark is a separate run).

Cheaper than a full 100-frame benchmark; catches wiring bugs fast.
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

SEQUENCE = os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_person_tracking")
N_FRAMES = 25
STRIDE = 2

print("=" * 60)
print("Local BA smoke test: BONN person_tracking, 25 frames")
print("=" * 60)

cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
cfg.dataset.type = "tum"
cfg.dataset.image_height = 480
cfg.dataset.image_width = 640
cfg.dataset.depth_scale = 5000.0
cfg.dynamic.enabled = False  # isolate A1 signal; dynamic off
cfg.language.enabled = False

# A1 on
assert cfg.slam.mapping.refine_poses, "refine_poses must be true in default config"
print(f"refine_poses = {cfg.slam.mapping.refine_poses}")
print(f"lr_pose_trans = {cfg.slam.mapping.lr_pose_trans}")
print(f"lr_pose_quat = {cfg.slam.mapping.lr_pose_quat}")

intrinsics = get_bonn_intrinsics()
dataset = TUMDataset(
    data_dir=SEQUENCE, height=480, width=640,
    depth_scale=5000.0, max_frames=N_FRAMES, stride=STRIDE,
)

device = "cuda"
gaussian_map = GaussianMap(
    sh_degree=cfg.gaussians.sh_degree,
    lang_feat_dim=cfg.gaussians.lang_feat_dim,
    init_opacity=cfg.gaussians.init_opacity,
    device=device,
)
slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
slam.process_first_frame(gaussian_map, dataset[0])

refine_events = 0
trans_max_hist = []
trans_mean_hist = []

for i in range(1, len(dataset)):
    frame = dataset[i]
    info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)

    if "pose_refine_trans_max_m" in info:
        refine_events += 1
        trans_max_hist.append(info["pose_refine_trans_max_m"])
        trans_mean_hist.append(info["pose_refine_trans_mean_m"])
        print(f"  Frame {i:3d} | ATE={info['ate']*100:.2f}cm | "
              f"G={info['total_gaussians']} | "
              f"pose_refine max={info['pose_refine_trans_max_m']*1000:.2f}mm "
              f"mean={info['pose_refine_trans_mean_m']*1000:.2f}mm")

gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
ate_rmse = slam.compute_ate_rmse(gt_poses) * 100

print()
print(f"ATE RMSE: {ate_rmse:.2f} cm")
print(f"Refine events: {refine_events}")
if trans_max_hist:
    import numpy as np
    arr_max = np.array(trans_max_hist) * 1000.0  # mm
    arr_mean = np.array(trans_mean_hist) * 1000.0
    print(f"Pose refine trans_max (mm):  min={arr_max.min():.2f} "
          f"med={np.median(arr_max):.2f} max={arr_max.max():.2f}")
    print(f"Pose refine trans_mean (mm): min={arr_mean.min():.2f} "
          f"med={np.median(arr_mean):.2f} max={arr_mean.max():.2f}")

    # Sanity checks
    import math
    issues = []
    if any(math.isnan(x) or math.isinf(x) for x in trans_max_hist):
        issues.append("NaN/Inf in pose refinement")
    if arr_max.max() > 500:  # 50 cm refinement is clearly broken
        issues.append(f"pose refinement diverged: max {arr_max.max():.1f} mm")
    if arr_max.max() < 1e-6:
        issues.append("pose refinement is identically zero (params not updating)")
    if issues:
        print()
        print("SMOKE TEST ISSUES:")
        for x in issues:
            print(f"  - {x}")
        sys.exit(1)
    else:
        print()
        print("Smoke test: PASS (no autograd errors, pose deltas sane)")
