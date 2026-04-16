"""Test dynamic object detection on BONN RGB-D Dynamic dataset.

Runs YOLOv8 + temporal filter on real frames with a walking person.
Saves visualizations to results/bonn_dynamic_test/.
"""

import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from PIL import Image

from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.dynamic import DynamicDetector, TemporalFilter

# ---- Config ----
SEQUENCE = os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_person_tracking")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "bonn_dynamic_test")
YOLO_MODEL = "yolov8n-seg"  # lightweight; use yolov8x-seg for full pipeline
STRIDE = 10  # process every 10th frame for speed
MAX_FRAMES = 50  # limit for quick test

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("BONN RGB-D Dynamic — Detection Test")
print("=" * 60)

# ---- Load dataset ----
print(f"\nLoading dataset: {SEQUENCE}")
dataset = TUMDataset(SEQUENCE, stride=STRIDE, max_frames=MAX_FRAMES)
intrinsics = get_bonn_intrinsics()
print(f"  Intrinsics: fx={intrinsics['fx']}, fy={intrinsics['fy']}")

# ---- Init detector ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nInitializing YOLOv8 on {device}...")
detector = DynamicDetector(
    model_name=YOLO_MODEL,
    confidence_thresh=0.5,
    device=device,
)

temporal_filter = TemporalFilter(
    window_size=3,
    min_detections=2,
    dilation_kernel=5,
)

# ---- Process frames ----
print(f"\nProcessing {len(dataset)} frames...")
detections_per_frame = []
dynamic_pcts = []

for i in range(len(dataset)):
    sample = dataset[i]
    # Convert (3,H,W) [0,1] float -> (H,W,3) uint8 numpy
    rgb_np = (sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    t0 = time.time()
    raw_mask = detector.detect_and_merge(rgb_np)
    dt_detect = time.time() - t0

    # Temporal filter
    static_mask = temporal_filter.update(raw_mask)
    dynamic_pct = (static_mask < 0.5).sum().item() / static_mask.numel() * 100

    # Count detections
    detections = detector.detect(rgb_np)
    n_det = len(detections)
    det_classes = [d["class_name"] for d in detections]

    detections_per_frame.append(n_det)
    dynamic_pcts.append(dynamic_pct)

    print(f"  Frame {i:3d} | detect={dt_detect:.2f}s | "
          f"raw_dets={n_det} {det_classes} | "
          f"dynamic={dynamic_pct:.1f}%")

    # Save visualization for selected frames
    if i in [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]:
        vis = rgb_np.copy().astype(np.float32)
        dynamic_region = (static_mask < 0.5).numpy()
        if dynamic_region.any():
            vis[dynamic_region] = vis[dynamic_region] * 0.3 + np.array([255, 0, 0]) * 0.7
        # Also overlay raw detection in blue (before temporal filter)
        raw_dynamic = raw_mask.numpy()
        if raw_dynamic.any():
            raw_only = raw_dynamic & ~dynamic_region
            vis[raw_only] = vis[raw_only] * 0.5 + np.array([0, 0, 255]) * 0.5
        Image.fromarray(vis.clip(0, 255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
        )

# ---- Summary ----
total_dets = sum(detections_per_frame)
frames_with_dets = sum(1 for d in detections_per_frame if d > 0)
avg_dynamic = np.mean(dynamic_pcts)

print(f"\n{'='*60}")
print("BONN Dynamic Detection Summary:")
print(f"  Sequence: person_tracking")
print(f"  Frames processed: {len(dataset)}")
print(f"  Frames with detections: {frames_with_dets}/{len(dataset)}")
print(f"  Total detections: {total_dets}")
print(f"  Avg dynamic area: {avg_dynamic:.1f}%")
print(f"  Results saved to {OUTPUT_DIR}/")
print(f"{'='*60}")

if frames_with_dets > 0:
    print("\nSUCCESS: YOLOv8 detects dynamic objects in BONN sequences!")
else:
    print("\nWARNING: No detections found. Person may not be visible in sampled frames.")
