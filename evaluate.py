"""Day 3: Evaluate SLAM across multiple Replica scenes with rendering quality metrics.

Usage:
    python evaluate.py                              # Test all scenes, 200 frames
    python evaluate.py --scenes room0 room1 office0 # Specific scenes
    python evaluate.py --frames 100                 # Quick test
    python evaluate.py --tune                       # Hyperparameter sweep
"""

import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.core.renderer import GaussianRenderer
from dynlang_slam.slam.pipeline import SLAMPipeline


# ── Rendering quality metrics ──────────────────────────────────────────

def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 50.0
    return float(10 * np.log10(1.0 / mse))


def compute_ssim_np(pred: np.ndarray, gt: np.ndarray, window_size: int = 11) -> float:
    """Structural Similarity Index (numpy, per-channel average)."""
    from scipy.ndimage import uniform_filter
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_vals = []
    for c in range(pred.shape[2]):
        p, g = pred[:, :, c].astype(np.float64), gt[:, :, c].astype(np.float64)
        mu_p = uniform_filter(p, window_size)
        mu_g = uniform_filter(g, window_size)
        sigma_p2 = uniform_filter(p * p, window_size) - mu_p * mu_p
        sigma_g2 = uniform_filter(g * g, window_size) - mu_g * mu_g
        sigma_pg = uniform_filter(p * g, window_size) - mu_p * mu_g
        num = (2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)
        den = (mu_p ** 2 + mu_g ** 2 + C1) * (sigma_p2 + sigma_g2 + C2)
        ssim_vals.append(np.mean(num / den))
    return float(np.mean(ssim_vals))


def compute_depth_l1(pred_depth: np.ndarray, gt_depth: np.ndarray) -> float:
    """L1 depth error in meters (only where GT > 0)."""
    valid = gt_depth > 0
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(pred_depth[valid] - gt_depth[valid])))


# ── Run SLAM on a single scene ────────────────────────────────────────

def run_scene(scene: str, max_frames: int, cfg_overrides: dict = None,
              config_path: str = "configs/default.yaml", device: str = "cuda") -> dict:
    """Run SLAM on a single scene and return all metrics."""

    cfg = load_config(config_path, [])
    cfg.dataset.scene = scene
    cfg.dataset.max_frames = max_frames

    # Apply overrides
    if cfg_overrides:
        for key, val in cfg_overrides.items():
            parts = key.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)

    # Load dataset
    dataset_path = Path(cfg.dataset.path) / scene
    if not dataset_path.exists():
        return {"scene": scene, "error": f"Scene not found: {dataset_path}"}

    dataset = ReplicaDataset(
        data_dir=str(dataset_path),
        height=cfg.dataset.image_height,
        width=cfg.dataset.image_width,
        depth_scale=cfg.dataset.depth_scale,
        max_frames=max_frames,
    )
    intrinsics = get_replica_intrinsics(
        fx=cfg.camera.fx, fy=cfg.camera.fy,
        cx=cfg.camera.cx, cy=cfg.camera.cy,
        height=cfg.dataset.image_height,
        width=cfg.dataset.image_width,
    )

    # Init
    gaussian_map = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
    renderer = GaussianRenderer(near=0.01, far=100.0)
    K = intrinsics["K"].to(device)

    # Process first frame
    first_frame = dataset[0]
    slam.process_first_frame(gaussian_map, first_frame)

    # SLAM loop
    ate_errors = []
    gt_poses = [first_frame["pose"].to(device)]
    t_start = time.time()

    for i in range(1, len(dataset)):
        frame = dataset[i]
        gt_poses.append(frame["pose"].to(device))
        info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
        ate_errors.append(info["ate"])

        if i % 50 == 0:
            print(f"    {scene}: Frame {i}/{len(dataset)-1} | ATE: {info['ate']*100:.2f}cm")

    total_time = time.time() - t_start

    # Compute rendering quality on a set of evaluation frames
    eval_frames = list(range(0, len(dataset), max(1, len(dataset) // 10)))  # ~10 eval frames
    psnr_vals, ssim_vals, depth_l1_vals = [], [], []

    with torch.no_grad():
        for idx in eval_frames:
            frame = dataset[idx]
            pose = slam.estimated_poses[idx] if idx < len(slam.estimated_poses) else frame["pose"].to(device)
            viewmat = torch.inverse(pose)
            result = renderer(gaussian_map, viewmat, K,
                              cfg.dataset.image_width, cfg.dataset.image_height, downscale=1)

            gt_rgb = frame["rgb"].permute(1, 2, 0).cpu().numpy()
            pred_rgb = result["rgb"].cpu().numpy().clip(0, 1)
            gt_depth = frame["depth"].squeeze(0).cpu().numpy()
            pred_depth = result["depth"].squeeze(-1).cpu().numpy()

            psnr_vals.append(compute_psnr(pred_rgb, gt_rgb))
            try:
                ssim_vals.append(compute_ssim_np(pred_rgb, gt_rgb))
            except ImportError:
                ssim_vals.append(-1.0)
            depth_l1_vals.append(compute_depth_l1(pred_depth, gt_depth))

    # ATE stats
    ate_np = np.array(ate_errors) * 100  # cm
    ate_rmse = slam.compute_ate_rmse(gt_poses) * 100  # cm

    metrics = {
        "scene": scene,
        "frames": len(dataset),
        "total_gaussians": gaussian_map.num_gaussians,
        # Tracking
        "ate_rmse_cm": round(ate_rmse, 3),
        "ate_mean_cm": round(float(ate_np.mean()), 3),
        "ate_median_cm": round(float(np.median(ate_np)), 3),
        "ate_max_cm": round(float(ate_np.max()), 3),
        # Rendering
        "psnr": round(float(np.mean(psnr_vals)), 2),
        "ssim": round(float(np.mean(ssim_vals)), 4),
        "depth_l1_m": round(float(np.mean(depth_l1_vals)), 4),
        # Performance
        "total_time_s": round(total_time, 1),
        "fps": round(len(dataset) / total_time, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
    }

    # Reset peak memory for next scene
    torch.cuda.reset_peak_memory_stats()

    return metrics


# ── Hyperparameter sweep ──────────────────────────────────────────────

def run_sweep(scene: str, max_frames: int):
    """Sweep key hyperparameters on a single scene."""
    configs = [
        {"name": "baseline",       "overrides": {}},
        {"name": "lr_pose=0.005",  "overrides": {"slam.tracking.lr_pose": 0.005}},
        {"name": "lr_pose=0.001",  "overrides": {"slam.tracking.lr_pose": 0.001}},
        {"name": "lr_pose=0.0005", "overrides": {"slam.tracking.lr_pose": 0.0005}},
        {"name": "track_iters=30", "overrides": {"slam.tracking.iterations": 30}},
        {"name": "track_iters=80", "overrides": {"slam.tracking.iterations": 80}},
        {"name": "map_iters=30",   "overrides": {"slam.mapping.iterations": 30}},
        {"name": "map_iters=100",  "overrides": {"slam.mapping.iterations": 100}},
        {"name": "kf_every=3",     "overrides": {"slam.keyframe.every_n_frames": 3}},
        {"name": "kf_every=10",    "overrides": {"slam.keyframe.every_n_frames": 10}},
        {"name": "depth_w=0.5",    "overrides": {"loss.depth_weight": 0.5}},
        {"name": "depth_w=2.0",    "overrides": {"loss.depth_weight": 2.0}},
    ]

    print(f"\n{'='*80}")
    print(f" Hyperparameter Sweep on {scene} ({max_frames} frames)")
    print(f"{'='*80}\n")

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {config['name']}")
        metrics = run_scene(scene, max_frames, cfg_overrides=config["overrides"])
        metrics["config_name"] = config["name"]
        results.append(metrics)
        print(f"  -> ATE: {metrics['ate_rmse_cm']:.2f}cm | PSNR: {metrics['psnr']:.1f}dB | Time: {metrics['total_time_s']:.0f}s")

    # Sort by ATE
    results.sort(key=lambda x: x.get("ate_rmse_cm", 999))

    print(f"\n{'='*80}")
    print(f" Sweep Results (sorted by ATE RMSE)")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'ATE(cm)':>8} {'PSNR':>7} {'SSIM':>7} {'Time(s)':>8} {'Gaussians':>10}")
    print(f"{'-'*20} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    for r in results:
        print(f"{r.get('config_name','?'):<20} {r['ate_rmse_cm']:>8.2f} {r['psnr']:>7.1f} {r['ssim']:>7.4f} {r['total_time_s']:>8.1f} {r['total_gaussians']:>10}")

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DynLang-SLAM")
    parser.add_argument("--scenes", nargs="+",
                        default=["room0", "room1", "room2", "office0", "office1"],
                        help="Scenes to evaluate")
    parser.add_argument("--frames", type=int, default=200, help="Max frames per scene")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--tune-scene", type=str, default="room0", help="Scene for tuning")
    args = parser.parse_args()

    if args.tune:
        results = run_sweep(args.tune_scene, args.frames)
        out_path = Path("results") / "sweep_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
        return

    # Multi-scene evaluation
    print(f"\n{'='*80}")
    print(f" DynLang-SLAM Evaluation — {len(args.scenes)} scenes, {args.frames} frames each")
    print(f"{'='*80}\n")

    all_results = []
    for scene in args.scenes:
        print(f"\n--- {scene} ---")
        metrics = run_scene(scene, args.frames)
        all_results.append(metrics)

        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
            continue

        print(f"  ATE RMSE:  {metrics['ate_rmse_cm']:.3f} cm")
        print(f"  PSNR:      {metrics['psnr']:.2f} dB")
        print(f"  SSIM:      {metrics['ssim']:.4f}")
        print(f"  Depth L1:  {metrics['depth_l1_m']:.4f} m")
        print(f"  Gaussians: {metrics['total_gaussians']}")
        print(f"  Time:      {metrics['total_time_s']:.1f}s ({metrics['fps']:.2f} FPS)")

    # Summary table
    valid = [r for r in all_results if "error" not in r]
    if valid:
        print(f"\n{'='*90}")
        print(f" Summary")
        print(f"{'='*90}")
        print(f"{'Scene':<12} {'ATE(cm)':>8} {'PSNR(dB)':>9} {'SSIM':>7} {'DepthL1(m)':>11} {'Gaussians':>10} {'FPS':>6}")
        print(f"{'-'*12} {'-'*8} {'-'*9} {'-'*7} {'-'*11} {'-'*10} {'-'*6}")
        for r in valid:
            print(f"{r['scene']:<12} {r['ate_rmse_cm']:>8.2f} {r['psnr']:>9.2f} {r['ssim']:>7.4f} {r['depth_l1_m']:>11.4f} {r['total_gaussians']:>10} {r['fps']:>6.2f}")

        # Averages
        avg_ate = np.mean([r["ate_rmse_cm"] for r in valid])
        avg_psnr = np.mean([r["psnr"] for r in valid])
        avg_ssim = np.mean([r["ssim"] for r in valid])
        avg_depth = np.mean([r["depth_l1_m"] for r in valid])
        print(f"{'-'*12} {'-'*8} {'-'*9} {'-'*7} {'-'*11} {'-'*10} {'-'*6}")
        print(f"{'AVERAGE':<12} {avg_ate:>8.2f} {avg_psnr:>9.2f} {avg_ssim:>7.4f} {avg_depth:>11.4f}")
        print(f"{'='*90}")

    # Save results
    out_path = Path("results") / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
