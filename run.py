"""DynLang-SLAM main entry point.

Usage:
    python run.py                                    # Default config (room0, full SLAM)
    python run.py --config configs/default.yaml      # Custom config
    python run.py dataset.scene=room1                # Override scene
    python run.py dataset.max_frames=100             # Quick test with 100 frames
    python run.py --gt-pose                          # Use GT poses (mapping only)
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from dynlang_slam.utils.config import load_config, print_config
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline


def main():
    parser = argparse.ArgumentParser(description="DynLang-SLAM")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--gt-pose", action="store_true", help="Use ground truth poses (skip tracking)")
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, args.overrides)
    print_config(cfg)

    # Set device
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Set seed
    torch.manual_seed(cfg.training.seed)
    if device == "cuda":
        torch.cuda.manual_seed(cfg.training.seed)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset_path = Path(cfg.dataset.path) / cfg.dataset.scene
    dataset = ReplicaDataset(
        data_dir=str(dataset_path),
        height=cfg.dataset.image_height,
        width=cfg.dataset.image_width,
        depth_scale=cfg.dataset.depth_scale,
        max_frames=cfg.dataset.max_frames,
    )
    intrinsics = get_replica_intrinsics(
        fx=cfg.camera.fx, fy=cfg.camera.fy,
        cx=cfg.camera.cx, cy=cfg.camera.cy,
        height=cfg.dataset.image_height,
        width=cfg.dataset.image_width,
    )

    # Initialize Gaussian map
    print("\n[2/4] Initializing Gaussian map...")
    gaussian_map = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )

    # Initialize SLAM pipeline
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)

    # Process first frame (initialize map)
    first_frame = dataset[0]
    slam.process_first_frame(gaussian_map, first_frame)
    print(f"  Map initialized with {gaussian_map.num_gaussians} Gaussians")
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / (1024**3)
        print(f"  GPU memory: {mem:.2f} GB")

    # SLAM loop
    print(f"\n[3/4] Running SLAM on {len(dataset)} frames...")
    print(f"  Mode: {'GT poses (mapping only)' if args.gt_pose else 'Full SLAM (tracking + mapping)'}")
    print(f"  Tracking iters: {cfg.slam.tracking.iterations}")
    print(f"  Mapping iters: {cfg.slam.mapping.iterations}")
    print(f"  Keyframe every: {cfg.slam.keyframe.every_n_frames} frames")
    print()

    gt_poses = []
    ate_errors = []
    total_time = 0.0
    log_interval = cfg.logging.log_interval

    for i in range(1, len(dataset)):
        frame = dataset[i]
        gt_poses.append(frame["pose"].to(device))

        t0 = time.time()
        info = slam.process_frame(
            gaussian_map=gaussian_map,
            frame=frame,
            use_gt_pose=args.gt_pose,
        )
        frame_time = time.time() - t0
        total_time += frame_time

        ate_errors.append(info["ate"])

        # Log progress
        if i % log_interval == 0 or i == len(dataset) - 1:
            avg_ate = np.mean(ate_errors[-log_interval:]) * 100  # to cm
            fps = log_interval / (total_time + 1e-8) if i == log_interval else 1.0 / (frame_time + 1e-8)
            mem = torch.cuda.memory_allocated() / (1024**3) if device == "cuda" else 0

            status = "KF" if info.get("is_keyframe") else "  "
            kf_info = ""
            if info.get("is_keyframe"):
                kf_info = f" | +{info.get('gaussians_added', 0)} -{info.get('gaussians_pruned', 0)} gaussians"

            print(
                f"  [{status}] Frame {i:5d}/{len(dataset)-1} | "
                f"ATE: {info['ate']*100:6.2f}cm (avg: {avg_ate:5.2f}cm) | "
                f"Gaussians: {info['total_gaussians']:7d} | "
                f"Time: {frame_time:.2f}s | "
                f"VRAM: {mem:.1f}GB"
                f"{kf_info}"
            )

        # Save checkpoint
        if (
            cfg.logging.save_checkpoints
            and i > 0
            and i % cfg.logging.checkpoint_interval == 0
        ):
            ckpt_path = Path(cfg.logging.log_dir) / cfg.dataset.scene / f"checkpoint_{i:06d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "frame_id": i,
                "gaussian_map": gaussian_map.state_dict_compact(),
                "estimated_poses": [p.cpu() for p in slam.estimated_poses],
            }, str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final evaluation
    print(f"\n[4/4] Results Summary")
    print("=" * 60)

    # ATE RMSE
    all_gt_poses = [dataset[0]["pose"].to(device)] + gt_poses
    ate_rmse = slam.compute_ate_rmse(all_gt_poses)
    ate_errors_np = np.array(ate_errors) * 100  # convert to cm

    print(f"  Scene: {cfg.dataset.scene}")
    print(f"  Frames processed: {len(dataset)}")
    print(f"  Final Gaussians: {gaussian_map.num_gaussians}")
    print(f"  ATE RMSE: {ate_rmse*100:.3f} cm")
    print(f"  ATE Mean: {ate_errors_np.mean():.3f} cm")
    print(f"  ATE Median: {np.median(ate_errors_np):.3f} cm")
    print(f"  ATE Max: {ate_errors_np.max():.3f} cm")
    print(f"  Total time: {total_time:.1f}s ({len(dataset)/total_time:.1f} FPS)")
    if device == "cuda":
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")
    print("=" * 60)

    # Save final results
    results_dir = Path(cfg.logging.log_dir) / cfg.dataset.scene
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save estimated trajectory
    traj_path = results_dir / "estimated_trajectory.txt"
    with open(traj_path, "w") as f:
        for pose in slam.estimated_poses:
            pose_np = pose.cpu().numpy().flatten()
            f.write(" ".join(f"{v:.10e}" for v in pose_np) + "\n")
    print(f"\n  Trajectory saved: {traj_path}")

    # Save metrics
    metrics_path = results_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"scene: {cfg.dataset.scene}\n")
        f.write(f"frames: {len(dataset)}\n")
        f.write(f"gaussians: {gaussian_map.num_gaussians}\n")
        f.write(f"ate_rmse_cm: {ate_rmse*100:.4f}\n")
        f.write(f"ate_mean_cm: {ate_errors_np.mean():.4f}\n")
        f.write(f"ate_median_cm: {np.median(ate_errors_np):.4f}\n")
        f.write(f"ate_max_cm: {ate_errors_np.max():.4f}\n")
        f.write(f"total_time_s: {total_time:.2f}\n")
        f.write(f"fps: {len(dataset)/total_time:.2f}\n")
    print(f"  Metrics saved: {metrics_path}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
