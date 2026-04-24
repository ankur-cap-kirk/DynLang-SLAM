"""H3 safety benchmark: Replica room0, 100 frames, stride=1, full pipeline.

Pre-registered in research/experiments/tracking-retry/protocol.md.
- Accept if ATE ≤ 1.50 cm
- Reject  if ATE > 2.00 cm (static-scene regression auto-rejects A2)
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

print("=" * 60)
print(f"H3 SAFETY: Replica {SCENE} ({N_FRAMES} frames, stride=1)")
print("  Pre-reg: accept<=1.50cm, reject>2.00cm")
print("=" * 60)

cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
cfg.dataset.scene = SCENE
cfg.dataset.max_frames = N_FRAMES
# Replica is static: dynamic masking / language turned on to match protocol
# "full pipeline" — but Replica has no dynamic classes so YOLO -> 0% masked.
cfg.dynamic.enabled = True
cfg.dynamic.yolo_model = "yolov8n-seg"
cfg.language.enabled = True
cfg.language.extract_every_n = 5
cfg.language.autoencoder.warmup_frames = 30

print(f"use_tracking_retry      = {cfg.slam.tracking.use_tracking_retry}")
print(f"retry_loss_ratio_thresh = {cfg.slam.tracking.retry_loss_ratio_thresh}")
print(f"retry_num_hypotheses    = {cfg.slam.tracking.retry_num_hypotheses}")

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
    height=cfg.dataset.image_height, width=cfg.dataset.image_width,
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

t0 = time.time()
retry_fired_frames: list[int] = []
for i in range(1, len(dataset)):
    frame = dataset[i]
    info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
    if info.get("retry_fired", False):
        retry_fired_frames.append(i)
    if i % 10 == 0 or i == len(dataset) - 1:
        print(f"  Frame {i:3d}/{len(dataset)-1} | ATE={info['ate']*100:.2f}cm | "
              f"G={info['total_gaussians']} | retry_fired={info.get('retry_fired', False)}")

total_time = time.time() - t0
ate = slam.compute_ate_rmse([dataset[i]["pose"].to(device) for i in range(len(dataset))]) * 100
print()
print(f"Final ATE RMSE (Replica {SCENE}, {N_FRAMES} frames): {ate:.2f} cm")
print(f"Retry fired on {len(retry_fired_frames)} / {len(dataset)-1} frames: {retry_fired_frames[:20]}")
print(f"Wall-clock: {total_time:.1f}s ({total_time/len(dataset):.2f}s/frame)")

if ate <= 1.50:
    print(f"H3 VERDICT: ACCEPT ({ate:.2f} <= 1.50)")
elif ate <= 2.00:
    print(f"H3 VERDICT: PASS (no reject) ({ate:.2f} <= 2.00)")
else:
    print(f"H3 VERDICT: REJECT ({ate:.2f} > 2.00)  -- A2 auto-rejects")
    sys.exit(1)
