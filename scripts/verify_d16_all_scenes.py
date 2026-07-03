"""Verify d=16 + warmup=20 + threshold=0.70 works on H2 and H3 too.

Only runs FULL mode (skip static / dynamic — those are not affected by
the language settings). Reports the same go/no-go signals as
test_d16_check.py:
  - Did the AE freeze?
  - At what frame?
  - Final language non-zero %
  - ATE-RMSE
"""
import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline


SCENES = [
    dict(tag="H2", kind="bonn", subdir="rgbd_bonn_balloon"),
    dict(tag="H3", kind="replica", subdir="room0"),
]
N_FRAMES = 100
STRIDE = 2  # for BONN; Replica uses stride 1
TARGET_DIM = 16


def run_one(scene):
    print("=" * 60)
    print(f"d={TARGET_DIM} viability test: {scene['tag']} {scene['subdir']}, "
          f"{N_FRAMES} frames, FULL mode")
    print("=" * 60)

    cfg = load_config(os.path.join(PROJECT_ROOT, "configs",
                                   "default.yaml"), [])

    if scene["kind"] == "bonn":
        seq_path = os.path.join(PROJECT_ROOT, "data", "BONN",
                                scene["subdir"])
        cfg.dataset.type = "tum"
        cfg.dataset.image_height = 480
        cfg.dataset.image_width = 640
        cfg.dataset.depth_scale = 5000.0
        intrinsics = get_bonn_intrinsics()
        dataset = TUMDataset(
            data_dir=seq_path,
            height=480, width=640,
            depth_scale=5000.0,
            max_frames=N_FRAMES,
            stride=STRIDE,
        )
    else:
        seq_path = os.path.join(PROJECT_ROOT, "data", "Replica",
                                scene["subdir"])
        cfg.dataset.type = "replica"
        cfg.dataset.image_height = 680
        cfg.dataset.image_width = 1200
        cfg.dataset.depth_scale = 6553.5
        intrinsics = get_replica_intrinsics()
        dataset = ReplicaDataset(
            data_dir=seq_path,
            height=680, width=1200,
            depth_scale=6553.5,
            max_frames=N_FRAMES,
            stride=1,
        )

    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.language.enabled = True
    cfg.language.extract_every_n = 2
    cfg.language.scales = ["whole"]
    cfg.language.sam_checkpoint = os.path.join(
        PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
    cfg.gaussians.lang_feat_dim = TARGET_DIM
    cfg.language.autoencoder.latent_dim = TARGET_DIM
    cfg.language.autoencoder.warmup_frames = 20

    device = "cuda"
    gaussian_map = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)

    t_start = time.time()
    slam.process_first_frame(gaussian_map, dataset[0])

    ae_frozen_at = None
    for i in range(1, len(dataset)):
        info = slam.process_frame(gaussian_map, dataset[i],
                                  use_gt_pose=False)
        if (ae_frozen_at is None and slam._autoencoder is not None
                and getattr(slam._autoencoder, "is_frozen", False)):
            ae_frozen_at = i
            print(f"  >>> AE FROZE at frame {i}")
        if i % 20 == 0 or i == len(dataset) - 1:
            print(f"  Frame {i:3d}/{len(dataset)-1} | "
                  f"ATE={info['ate']*100:.2f}cm | G={info['total_gaussians']}")

    total_time = time.time() - t_start
    gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    ate_rmse = slam.compute_ate_rmse(gt_poses) * 100

    lang_feats = gaussian_map.lang_feats.data
    nonzero = (lang_feats.abs().sum(dim=-1) > 0.01).sum().item()
    nonzero_pct = nonzero / lang_feats.shape[0] * 100

    result = dict(
        scene=scene["tag"],
        ate_cm=ate_rmse,
        gaussians=gaussian_map.num_gaussians,
        ae_frozen=ae_frozen_at is not None,
        ae_frozen_at=ae_frozen_at,
        lang_nonzero_pct=nonzero_pct,
        total_time_s=total_time,
    )
    print(f"  {scene['tag']}: AE frozen={result['ae_frozen']} "
          f"@{ae_frozen_at}  lang={nonzero_pct:.1f}%  "
          f"ATE={ate_rmse:.2f}cm  time={total_time:.0f}s")

    del gaussian_map, slam
    torch.cuda.empty_cache()
    return result


def main():
    results = []
    for scene in SCENES:
        try:
            results.append(run_one(scene))
        except Exception as e:
            print(f"  {scene['tag']} FAILED: {type(e).__name__}: {e}")
            results.append(dict(scene=scene["tag"], error=str(e)))

    print("\n=== SUMMARY ===")
    print(f"{'scene':<5} {'AE frozen':<11} {'AE@':<5} "
          f"{'lang%':<7} {'ATE cm':<8} {'time s':<7}")
    for r in results:
        if "error" in r:
            print(f"{r['scene']:<5} ERROR: {r['error']}")
        else:
            print(f"{r['scene']:<5} {str(r['ae_frozen']):<11} "
                  f"{str(r['ae_frozen_at'] or '-'):<5} "
                  f"{r['lang_nonzero_pct']:<7.1f} {r['ate_cm']:<8.2f} "
                  f"{r['total_time_s']:<7.0f}")


if __name__ == "__main__":
    main()
