"""Export per-scene trajectories (estimated + GT) to NPZ for figure scripts.

Runs the baseline DynLang-SLAM pipeline once per scene per seed and saves
committed est + GT poses as compact NPZ files. Downstream figure scripts
load the NPZs without GPU, so plot iteration is fast.

Config matches br1_eval.py's BR1-off baseline (dynamic ON, language OFF,
BR1 off, A2 off, PIPs off).

Multi-seed rationale:
    The quoted "aligned_baselines_2026_04_23" (7.56 / 8.55 / 3.68) were
    single-run snapshots with no seed and no CUDA determinism flags.
    Re-running the same code today produces 13.52 / 10.12 / 4.64 (delta
    +5.96 / +1.57 / +0.96), i.e. the "baselines" were inside the noise
    band. For a reproducible paper, every reported number must be a
    MEAN OVER SEEDS with a stdev.

Usage:
    python scripts/export_trajectories.py                   # all 3 scenes, all 3 seeds
    DYNLANG_SCENES=h3 python scripts/export_trajectories.py      # one scene
    DYNLANG_SEEDS=42 python scripts/export_trajectories.py       # single-seed (fast)
    DYNLANG_SEEDS=42,123,2024 python scripts/export_trajectories.py  # explicit seed list

Outputs (under results/figures/):
    traj_h1_person_tracking_seed{S}.npz
    traj_h2_balloon_seed{S}.npz
    traj_h3_replica_room0_seed{S}.npz
    Each NPZ holds: est_c2w (N,4,4), gt_c2w (N,4,4), frame_ids (N,), seed (int)
"""
import os
import random
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Set BEFORE importing torch so cuBLAS reductions become deterministic-safe
# (PyTorch honors this only if set prior to first CUDA context init).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from dynlang_slam.utils.config import load_config
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline


SCENES = os.environ.get("DYNLANG_SCENES", "h1,h2,h3").split(",")
SEEDS = [int(s) for s in os.environ.get("DYNLANG_SEEDS", "42,123,2024").split(",")]
N_BONN = int(os.environ.get("DYNLANG_N_FRAMES_BONN", "100"))
STRIDE_BONN = int(os.environ.get("DYNLANG_STRIDE_BONN", "2"))
N_REPLICA = int(os.environ.get("DYNLANG_N_FRAMES_REPLICA", "100"))

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def _seed_everything(seed: int) -> None:
    """Seed every RNG we can reach + enable deterministic CUDA kernels.

    Note: gsplat rasterizer is still non-deterministic on some GPUs
    (atomicAdd in the backward pass). Determinism flags here lock down
    everything we can control (torch tensors, YOLO init, densification),
    which shrinks the noise band substantially but does not zero it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN: pick deterministic algorithms, disable autotuning.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Torch's own deterministic algorithms flag (errors noisily if a
    # non-deterministic op is hit -- useful signal, but warn_only keeps
    # the run from crashing on ops gsplat can't replace).
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _build_cfg_bonn():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = False
    cfg.language.enabled = False
    cfg.slam.bootstrap.enabled = False  # baseline, BR1 off
    return cfg


def _build_cfg_replica():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.scene = "room0"
    cfg.dataset.max_frames = N_REPLICA
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = False
    cfg.language.enabled = False
    cfg.slam.bootstrap.enabled = False
    return cfg


def run_bonn(scene_key: str, seq_name: str, seed: int):
    from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics

    print(f"\n{'='*72}")
    print(f" [export] {scene_key}  {seq_name}  n={N_BONN} stride={STRIDE_BONN}  seed={seed}")
    print(f"{'='*72}")
    _seed_everything(seed)

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
    dt = time.time() - t0

    # Collect trajectories (same order process_frame stored them in)
    n = min(len(slam.estimated_poses), len(dataset))
    est = np.stack([
        slam.estimated_poses[i].detach().cpu().numpy() for i in range(n)
    ], axis=0)  # (N, 4, 4)
    gt = np.stack([
        dataset[i]["pose"].detach().cpu().numpy() if torch.is_tensor(dataset[i]["pose"])
        else np.asarray(dataset[i]["pose"])
        for i in range(n)
    ], axis=0)  # (N, 4, 4)

    suffix = f"{seq_name.replace('rgbd_bonn_', '')}_seed{seed}"
    out_path = os.path.join(OUT_DIR, f"traj_{scene_key}_{suffix}.npz")
    np.savez_compressed(
        out_path,
        est_c2w=est,
        gt_c2w=gt,
        frame_ids=np.arange(n),
        scene=scene_key,
        sequence=seq_name,
        seed=seed,
        runtime_s=dt,
    )
    print(f"  wrote {out_path}   (N={n} poses, runtime={dt:.1f}s)")

    del gmap, slam
    torch.cuda.empty_cache()


def run_replica(scene_key: str, seed: int):
    from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
    from pathlib import Path

    print(f"\n{'='*72}")
    print(f" [export] {scene_key}  replica_room0  n={N_REPLICA}  seed={seed}")
    print(f"{'='*72}")
    _seed_everything(seed)

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
    dt = time.time() - t0

    n = min(len(slam.estimated_poses), len(dataset))
    est = np.stack([
        slam.estimated_poses[i].detach().cpu().numpy() for i in range(n)
    ], axis=0)
    gt = np.stack([
        dataset[i]["pose"].detach().cpu().numpy() if torch.is_tensor(dataset[i]["pose"])
        else np.asarray(dataset[i]["pose"])
        for i in range(n)
    ], axis=0)

    out_path = os.path.join(OUT_DIR, f"traj_{scene_key}_replica_room0_seed{seed}.npz")
    np.savez_compressed(
        out_path,
        est_c2w=est,
        gt_c2w=gt,
        frame_ids=np.arange(n),
        scene=scene_key,
        sequence="replica_room0",
        seed=seed,
        runtime_s=dt,
    )
    print(f"  wrote {out_path}   (N={n} poses, runtime={dt:.1f}s)")

    del gmap, slam
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"[export] scenes={SCENES}  seeds={SEEDS}")
    print(f"[export] total runs = {len(SCENES) * len(SEEDS)}  "
          f"(expect ~{len(SCENES) * len(SEEDS) * 4} min wall-clock)")
    t_start = time.time()
    for seed in SEEDS:
        if "h1" in SCENES:
            run_bonn("h1", "rgbd_bonn_person_tracking", seed)
        if "h2" in SCENES:
            run_bonn("h2", "rgbd_bonn_balloon", seed)
        if "h3" in SCENES:
            run_replica("h3", seed)
    print(f"\n[done] trajectories saved under {OUT_DIR}  "
          f"(total wall-clock {(time.time()-t_start)/60:.1f} min)")
