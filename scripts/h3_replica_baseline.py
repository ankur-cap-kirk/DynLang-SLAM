"""H3 baseline sanity: Replica room0, 100 frames, stride=1, full pipeline
WITH use_tracking_retry=False.

Confirms whether the H3 regression (13.66 cm vs protocol-stated ~1.2 cm)
is caused by A2 or is a pre-existing pipeline state. retry_fired was False
for all 99 frames in the A2-enabled run, so A2 cannot be the proximate cause —
but we verify directly by disabling the knob.
"""
import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from pathlib import Path

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

N_FRAMES = 100
SCENE = "room0"

cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
cfg.dataset.scene = SCENE
cfg.dataset.max_frames = N_FRAMES
cfg.dynamic.enabled = True
cfg.dynamic.yolo_model = "yolov8n-seg"
cfg.language.enabled = True
cfg.language.extract_every_n = 5
cfg.language.autoencoder.warmup_frames = 30
# KEY: disable A2
cfg.slam.tracking.use_tracking_retry = False

print(f"use_tracking_retry = {cfg.slam.tracking.use_tracking_retry}  (baseline)")

dataset_path = Path(PROJECT_ROOT) / cfg.dataset.path / SCENE
dataset = ReplicaDataset(
    data_dir=str(dataset_path), height=cfg.dataset.image_height,
    width=cfg.dataset.image_width, depth_scale=cfg.dataset.depth_scale,
    max_frames=N_FRAMES,
)
intrinsics = get_replica_intrinsics(
    fx=cfg.camera.fx, fy=cfg.camera.fy, cx=cfg.camera.cx, cy=cfg.camera.cy,
    height=cfg.dataset.image_height, width=cfg.dataset.image_width,
)

device = "cuda"
gaussian_map = GaussianMap(
    sh_degree=cfg.gaussians.sh_degree, lang_feat_dim=cfg.gaussians.lang_feat_dim,
    init_opacity=cfg.gaussians.init_opacity, device=device,
)
slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
slam.process_first_frame(gaussian_map, dataset[0])

t0 = time.time()
for i in range(1, len(dataset)):
    frame = dataset[i]
    info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
    if i % 10 == 0 or i == len(dataset) - 1:
        print(f"  Frame {i:3d}/{len(dataset)-1} | ATE={info['ate']*100:.2f}cm | G={info['total_gaussians']}")

ate = slam.compute_ate_rmse([dataset[i]["pose"].to(device) for i in range(len(dataset))]) * 100
print(f"\nBaseline (A2 off) ATE RMSE: {ate:.2f} cm in {time.time()-t0:.1f}s")
