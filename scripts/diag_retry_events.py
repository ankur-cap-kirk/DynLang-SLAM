"""Diagnostic: count and log retry events over a 100-frame BONN run.

Run for both person_tracking and balloon to see whether the A2 loss-ratio
trigger is actually firing on the known drift event (frame 60-70 of
person_tracking). If retry never fires, the H1 +1.27cm regression is run-
to-run noise rather than a real A2 effect.
"""
import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

SEQ_NAME = os.environ.get("DYNLANG_SEQUENCE", "rgbd_bonn_person_tracking")
SEQUENCE = os.path.join(PROJECT_ROOT, "data", "BONN", SEQ_NAME)
N_FRAMES = 100
STRIDE = 2

print("=" * 60)
print(f"A2 diagnostic: {SEQ_NAME}, {N_FRAMES} frames stride={STRIDE}")
print("=" * 60)

cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
cfg.dataset.type = "tum"
cfg.dataset.image_height = 480
cfg.dataset.image_width = 640
cfg.dataset.depth_scale = 5000.0
cfg.dynamic.enabled = True
cfg.dynamic.yolo_model = "yolov8n-seg"
cfg.language.enabled = False  # skip language to be fast

print(f"use_tracking_retry      = {cfg.slam.tracking.use_tracking_retry}")
print(f"retry_loss_ratio_thresh = {cfg.slam.tracking.retry_loss_ratio_thresh}")

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

ate_hist: list[float] = []
loss_hist: list[float] = []
retry_events: list[dict] = []

t0 = time.time()
for i in range(1, len(dataset)):
    frame = dataset[i]
    info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
    ate_hist.append(info["ate"])
    loss_hist.append(info.get("tracking_loss", 0.0))
    if info.get("retry_fired", False):
        ev = {
            "frame": i,
            "trigger_ratio": info.get("retry_trigger_ratio"),
            "winner": info.get("retry_winner"),
            "losses": info.get("retry_losses", {}),
        }
        retry_events.append(ev)
        print(f"  [RETRY] frame {i:3d}: trig={ev['trigger_ratio']:.2f}x "
              f"winner={ev['winner']} losses={ev['losses']}")
    if i % 10 == 0 or i == len(dataset) - 1:
        print(f"  Frame {i:3d}/{len(dataset)-1} | ATE={info['ate']*100:.2f}cm | "
              f"loss={info.get('tracking_loss', 0):.4f} | "
              f"G={info['total_gaussians']} | retry_total={len(retry_events)}")

gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
ate_rmse = slam.compute_ate_rmse(gt_poses) * 100
print()
print(f"Final ATE RMSE: {ate_rmse:.2f} cm in {time.time()-t0:.1f}s")
print(f"Retry events: {len(retry_events)} / {len(dataset)-1} frames")
if retry_events:
    wins = [e["winner"] for e in retry_events]
    primary_wins = wins.count("primary")
    print(f"  Winners: primary={primary_wins}, alternates={len(wins)-primary_wins}")
    alt_winners = [w for w in wins if w != "primary"]
    if alt_winners:
        from collections import Counter
        print(f"  Alt breakdown: {dict(Counter(alt_winners))}")
if loss_hist:
    arr = np.array(loss_hist)
    print(f"Loss: min={arr.min():.4f} med={np.median(arr):.4f} max={arr.max():.4f} "
          f"max/med={arr.max()/np.median(arr):.2f}x")
