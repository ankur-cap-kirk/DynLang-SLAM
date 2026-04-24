"""Re-baseline H1/H2/H3 under Umeyama-aligned ATE.

After discovering compute_ate_rmse was reporting RAW translation residuals,
all pre-registered budgets and baselines are now suspect. This script
re-measures the three safety-gate scenes in full-pipeline mode
(matching the numbers stored in research-state.yaml as 'full' ATE) and
prints BOTH raw and aligned ATE for each.

Pick scenes via env var, default = all three:

    DYNLANG_SCENES=h1           # just person_tracking
    DYNLANG_SCENES=h1,h2        # both BONN scenes
    DYNLANG_SCENES=h1,h2,h3     # full rebaseline
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
from pathlib import Path

from dynlang_slam.utils.config import load_config
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline


SCENES = os.environ.get("DYNLANG_SCENES", "h1,h2,h3").split(",")
N_BONN = int(os.environ.get("DYNLANG_N_FRAMES_BONN", "100"))
STRIDE_BONN = int(os.environ.get("DYNLANG_STRIDE_BONN", "2"))
N_REPLICA = int(os.environ.get("DYNLANG_N_FRAMES_REPLICA", "100"))

# Light mode: dynamic ON, language OFF. Keeps runtime tractable and matches
# the dynamic-only comparison point already verified with Umeyama. Set
# DYNLANG_FULL=1 to re-run with language enabled (matches state file's
# 'full' column exactly but takes ~3x longer).
USE_LANGUAGE = os.environ.get("DYNLANG_FULL", "0") == "1"


def _build_cfg_bonn():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.language.enabled = USE_LANGUAGE
    if USE_LANGUAGE:
        cfg.language.extract_every_n = 2
        cfg.language.autoencoder.warmup_frames = 30
        cfg.language.sam_checkpoint = os.path.join(
            PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"
        )
    return cfg


def _build_cfg_replica():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.scene = "room0"
    cfg.dataset.max_frames = N_REPLICA
    # Replica is static; dynamic enabled matches h3_replica_safety.py.
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.language.enabled = USE_LANGUAGE
    if USE_LANGUAGE:
        cfg.language.extract_every_n = 5
        cfg.language.autoencoder.warmup_frames = 30
        cfg.language.sam_checkpoint = os.path.join(
            PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"
        )
    return cfg


def run_bonn_scene(label: str, seq_name: str):
    from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
    print(f"\n{'='*64}\n {label}  BONN {seq_name}  n={N_BONN} stride={STRIDE_BONN}  "
          f"language={'ON' if USE_LANGUAGE else 'OFF'}\n{'='*64}")

    cfg = _build_cfg_bonn()
    dataset = TUMDataset(
        data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", seq_name),
        height=480, width=640, depth_scale=5000.0,
        max_frames=N_BONN, stride=STRIDE_BONN,
    )
    device = "cuda"
    gmap = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=get_bonn_intrinsics(), device=device)
    slam.process_first_frame(gmap, dataset[0])

    t0 = time.time()
    for i in range(1, len(dataset)):
        slam.process_frame(gmap, dataset[i], use_gt_pose=False)
    gt = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    raw = slam.compute_ate_rmse(gt, align=False) * 100
    aln = slam.compute_ate_rmse(gt, align=True) * 100
    dt = time.time() - t0
    print(f" runtime={dt:.1f}s")
    print(f" raw     ATE-RMSE: {raw:.2f} cm")
    print(f" ALIGNED ATE-RMSE: {aln:.2f} cm")
    # Free GPU before next scene
    del gmap, slam
    torch.cuda.empty_cache()
    return raw, aln


def run_replica_scene(label: str):
    from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
    print(f"\n{'='*64}\n {label}  Replica room0  n={N_REPLICA}  "
          f"language={'ON' if USE_LANGUAGE else 'OFF'}\n{'='*64}")

    cfg = _build_cfg_replica()
    dataset_path = Path(PROJECT_ROOT) / cfg.dataset.path / "room0"
    dataset = ReplicaDataset(
        data_dir=str(dataset_path),
        height=cfg.dataset.image_height,
        width=cfg.dataset.image_width,
        depth_scale=cfg.dataset.depth_scale,
        max_frames=N_REPLICA,
    )
    intrinsics = get_replica_intrinsics(
        fx=cfg.camera.fx, fy=cfg.camera.fy,
        cx=cfg.camera.cx, cy=cfg.camera.cy,
        height=cfg.dataset.image_height, width=cfg.dataset.image_width,
    )
    device = "cuda"
    gmap = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
    slam.process_first_frame(gmap, dataset[0])

    t0 = time.time()
    for i in range(1, len(dataset)):
        slam.process_frame(gmap, dataset[i], use_gt_pose=False)
    gt = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    raw = slam.compute_ate_rmse(gt, align=False) * 100
    aln = slam.compute_ate_rmse(gt, align=True) * 100
    dt = time.time() - t0
    print(f" runtime={dt:.1f}s")
    print(f" raw     ATE-RMSE: {raw:.2f} cm")
    print(f" ALIGNED ATE-RMSE: {aln:.2f} cm")
    del gmap, slam
    torch.cuda.empty_cache()
    return raw, aln


results = {}
if "h1" in SCENES:
    results["H1_person_tracking"] = run_bonn_scene("H1", "rgbd_bonn_person_tracking")
if "h2" in SCENES:
    results["H2_balloon"] = run_bonn_scene("H2", "rgbd_bonn_balloon")
if "h3" in SCENES:
    results["H3_replica_room0"] = run_replica_scene("H3")

print(f"\n{'='*64}\n SUMMARY — re-baseline under Umeyama-aligned ATE\n{'='*64}")
print(f" {'scene':<28s} {'raw (cm)':>10s} {'aligned (cm)':>14s} {'ratio':>8s}")
print(f" {'-'*28} {'-'*10} {'-'*14} {'-'*8}")
for name, (raw, aln) in results.items():
    ratio = raw / max(aln, 1e-6)
    print(f" {name:<28s} {raw:>10.2f} {aln:>14.2f} {ratio:>7.2f}x")

# Reference SOTA for context
print(f"\n SOTA ref (aligned): DG-SLAM=4.73cm, BDGS-SLAM=4.03cm")
print(f" Prior stored (raw): person_tracking=20.32, balloon=57.35, replica ~13.7")
