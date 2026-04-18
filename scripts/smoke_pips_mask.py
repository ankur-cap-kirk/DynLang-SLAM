"""Diagnostic H0 pre-check: does CoTracker add class-agnostic dynamic signal
on BONN balloon beyond what YOLO provides?

Accept H0 iff average pips_added_over_yolo_pct >= 0.5 pp across the test.

This runs BEFORE full ATE benchmark so we can kill intervention B' early if
the signal isn't informative in our regime.
"""
import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.dynamic import (
    DynamicDetector, TemporalFilter,
    PointTrackBuffer, compute_pips_dynamic_mask,
)

device = "cuda"

# ---- Config ----
SEQ = os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_balloon")
T = 8                 # CoTracker window
N_FRAMES = 30         # enough to fill buffer and sample residuals
STRIDE = 2
GRID = 20             # 400 query points
THRESH_PX = 5.0

print("="*60)
print("H0 DIAGNOSTIC: does PIPs add signal beyond YOLO on balloon?")
print("="*60)

# Load dataset
ds = TUMDataset(data_dir=SEQ, height=480, width=640, depth_scale=5000.0,
                max_frames=N_FRAMES, stride=STRIDE)
intr = get_bonn_intrinsics()
K = intr["K"].to(device)

# Load YOLO
yolo = DynamicDetector(
    model_name="yolov8n-seg",
    confidence_thresh=0.5,
    dynamic_classes=[0, 1, 2, 3, 5, 14, 15, 16, 17],
    device=device,
)
tf = TemporalFilter(window_size=3, min_detections=2, dilation_kernel=5)

# Load CoTracker
print("Loading CoTracker2...")
model = torch.hub.load("facebookresearch/co-tracker", "cotracker2",
                       trust_repo=True).to(device).eval()

# Loop through frames, push to buffer, after full compute mask, log diagnostics
buffer = PointTrackBuffer(window_size=T)

added_pcts = []
pips_pcts = []
yolo_pcts = []
runtimes_ms = []

for i in range(N_FRAMES):
    frame = ds[i]
    rgb = frame["rgb"].to(device)      # (3, H, W) or (H, W, 3)
    depth = frame["depth"].to(device)  # (H, W)
    gt_pose = frame["pose"].to(device) # (4, 4)

    # Use GT pose — this is a SIGNAL test, not an ATE test, so pose error is
    # not what we're probing. We want to know if PIPs trajectories disagree
    # with the rigid-motion prediction under KNOWN GOOD ego-motion.
    buffer.push(rgb, depth, gt_pose)

    # YOLO mask (1=static, 0=dynamic)
    rgb_chw = rgb if rgb.dim() == 3 and rgb.shape[0] == 3 else rgb.permute(2, 0, 1)
    yolo_dyn_bool = yolo.detect_and_merge(rgb_chw)   # (H, W) bool, True=dynamic
    yolo_static = (~yolo_dyn_bool).float()           # 1=static, 0=dynamic
    yolo_static_tf = tf.update(yolo_dyn_bool)        # applies temporal filter
    # temporal_filter returns 1=static float.
    yolo_static = yolo_static_tf

    if not buffer.is_full():
        print(f"Frame {i:2d}: filling buffer ({len(buffer)}/{T})")
        continue

    t0 = time.time()
    result = compute_pips_dynamic_mask(
        buffer=buffer, K=K, model=model,
        grid=GRID, residual_thresh_px=THRESH_PX,
        semantic_mask=yolo_static,
    )
    dt_ms = (time.time() - t0) * 1000
    if result is None:
        continue
    static_mask, info = result
    yolo_pct = 100.0 * (1.0 - yolo_static.float().mean().item())
    pips_pct = info["pips_dynamic_pct"]
    added = info["pips_added_over_yolo_pct"]
    print(f"Frame {i:2d}: yolo_dyn={yolo_pct:5.2f}%  pips_dyn={pips_pct:5.2f}%  "
          f"added={added:5.2f}%  med_res={info['pips_median_residual_px']:.2f}px  "
          f"max_res={info['pips_max_residual_px']:5.1f}px  "
          f"npts={info['pips_dyn_points']}/{info['pips_total_points']}  "
          f"t={dt_ms:.0f}ms")
    added_pcts.append(added)
    pips_pcts.append(pips_pct)
    yolo_pcts.append(yolo_pct)
    runtimes_ms.append(dt_ms)

print("="*60)
print("SUMMARY")
print("="*60)
if added_pcts:
    print(f"Mean YOLO dynamic%      : {np.mean(yolo_pcts):.2f}")
    print(f"Mean PIPs dynamic%      : {np.mean(pips_pcts):.2f}")
    print(f"Mean PIPs-added %       : {np.mean(added_pcts):.2f}")
    print(f"Median PIPs-added %     : {np.median(added_pcts):.2f}")
    print(f"Mean runtime            : {np.mean(runtimes_ms):.0f}ms/call")
    verdict = "PASS" if np.mean(added_pcts) >= 0.5 else "FAIL"
    print(f"\nH0 verdict (threshold 0.5 pp added): {verdict}")
    if verdict == "FAIL":
        print("  PIPs is not adding class-agnostic signal beyond YOLO on this")
        print("  sequence. Consider: (a) lowering threshold, (b) increasing grid,")
        print("  or (c) rejecting intervention B' on ineffectiveness.")
else:
    print("ERROR: no frames where buffer was full. Test didn't run.")
