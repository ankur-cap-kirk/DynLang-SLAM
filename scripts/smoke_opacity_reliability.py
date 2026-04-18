"""Smoke test for intervention D1 (opacity reliability mask + loss weight flip).

Runs 25 frames on BONN person_tracking, full pipeline. Verifies:
  1. No NaN/Inf in per-frame loss or ATE.
  2. Loss magnitude is finite and sane (not exploded, not collapsed to 0).
  3. Final 25-frame ATE within reason (< 50 cm) — not a full benchmark.
  4. reliability_thresh config value is picked up.

Cheaper than a full 100-frame benchmark; catches wiring bugs fast before
committing GPU time to H1/H2/H3.
"""
import os, sys, math
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

SEQUENCE = os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_person_tracking")
N_FRAMES = 25
STRIDE = 2

print("=" * 60)
print("D1 (opacity reliability + loss-weight flip) smoke test")
print(f"  BONN person_tracking, {N_FRAMES} frames stride={STRIDE}")
print("=" * 60)

cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
cfg.dataset.type = "tum"
cfg.dataset.image_height = 480
cfg.dataset.image_width = 640
cfg.dataset.depth_scale = 5000.0
cfg.dynamic.enabled = True
cfg.dynamic.yolo_model = "yolov8n-seg"
cfg.language.enabled = False  # keep smoke fast

# D1 assertions
assert not cfg.slam.mapping.refine_poses, "refine_poses must be false (A1 rejected)"
assert abs(cfg.loss.rgb_weight - 0.9) < 1e-6, f"rgb_weight must be 0.9, got {cfg.loss.rgb_weight}"
assert abs(cfg.loss.depth_weight - 0.1) < 1e-6, f"depth_weight must be 0.1, got {cfg.loss.depth_weight}"
assert abs(cfg.loss.reliability_thresh - 0.5) < 1e-6, f"reliability_thresh must be 0.5, got {cfg.loss.reliability_thresh}"

print(f"loss.rgb_weight       = {cfg.loss.rgb_weight}")
print(f"loss.depth_weight     = {cfg.loss.depth_weight}")
print(f"loss.ssim_weight      = {cfg.loss.ssim_weight}")
print(f"loss.reliability_thresh = {cfg.loss.reliability_thresh}")

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

ate_hist = []
loss_hist = []

for i in range(1, len(dataset)):
    frame = dataset[i]
    info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
    ate_hist.append(info["ate"])
    if "tracker_final_loss" in info:
        loss_hist.append(info["tracker_final_loss"])
    elif "final_loss" in info:
        loss_hist.append(info["final_loss"])

    if i % 5 == 0 or i == len(dataset) - 1:
        print(f"  Frame {i:3d}/{len(dataset)-1} | ATE={info['ate']*100:.2f}cm | "
              f"G={info['total_gaussians']} | dyn={info.get('dynamic_pct', 0):.1f}%")

gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
ate_rmse = slam.compute_ate_rmse(gt_poses) * 100

print()
print(f"Final ATE RMSE (25 frames): {ate_rmse:.2f} cm")
if loss_hist:
    arr = np.array(loss_hist)
    print(f"Tracker loss: min={arr.min():.4f} med={np.median(arr):.4f} max={arr.max():.4f}")

# Sanity checks
issues = []
if math.isnan(ate_rmse) or math.isinf(ate_rmse):
    issues.append(f"ATE RMSE is NaN/Inf: {ate_rmse}")
if ate_rmse > 50.0:  # 25-frame smoke shouldn't blow this high
    issues.append(f"ATE RMSE diverged: {ate_rmse:.2f} cm > 50 cm")
if any(math.isnan(x) or math.isinf(x) for x in ate_hist):
    issues.append("NaN/Inf in per-frame ATE history")
if loss_hist:
    if any(math.isnan(x) or math.isinf(x) for x in loss_hist):
        issues.append("NaN/Inf in tracker loss history")
    if np.array(loss_hist).max() < 1e-8:
        issues.append("Tracker loss collapsed to zero (mask killed all pixels?)")

if issues:
    print()
    print("SMOKE TEST ISSUES:")
    for x in issues:
        print(f"  - {x}")
    sys.exit(1)
else:
    print()
    print("Smoke test: PASS (no NaN/Inf, ATE within sane bound)")
