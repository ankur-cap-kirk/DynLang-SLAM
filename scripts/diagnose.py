"""Systematic diagnostic: per-frame analysis of tracking drift.

Captures per-frame metrics to pinpoint WHERE and WHY drift happens:
  - ATE (translation error)
  - Tracking loss (optimizer convergence)
  - Alpha coverage (% of pixels with valid Gaussian rendering)
  - Depth validity ratio (% of GT depth covered by map)
  - Camera motion speed (GT inter-frame displacement)
  - Gaussian count over time

Usage:
    python diagnose.py --scenes room0 room2       # Compare good vs bad
    python diagnose.py --scenes room1 --frames 200
"""

import argparse
import time
import json
from pathlib import Path

import torch
import numpy as np

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.core.renderer import GaussianRenderer
from dynlang_slam.slam.pipeline import SLAMPipeline


def diagnose_scene(scene: str, max_frames: int, device: str = "cuda") -> dict:
    """Run SLAM with full per-frame diagnostics."""

    cfg = load_config("configs/default.yaml", [])
    cfg.dataset.scene = scene
    cfg.dataset.max_frames = max_frames

    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)

    dataset_path = Path(cfg.dataset.path) / scene
    if not dataset_path.exists():
        print(f"  ERROR: {dataset_path} not found")
        return {"scene": scene, "error": "not found"}

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

    gaussian_map = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
    renderer = GaussianRenderer(near=0.01, far=100.0)
    K = intrinsics["K"].to(device)
    W = cfg.dataset.image_width
    H = cfg.dataset.image_height

    # Process first frame
    first_frame = dataset[0]
    slam.process_first_frame(gaussian_map, first_frame)

    # Per-frame diagnostic storage
    per_frame = []
    gt_poses = [first_frame["pose"].to(device)]
    prev_gt_pos = first_frame["pose"][:3, 3].numpy()

    print(f"  {'Frame':>6} {'ATE(cm)':>8} {'Loss':>8} {'Alpha%':>7} {'DepVal%':>8} "
          f"{'Motion(mm)':>10} {'Gaussians':>10} {'Keyframe':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for i in range(1, len(dataset)):
        frame = dataset[i]
        gt_pose = frame["pose"].to(device)
        gt_poses.append(gt_pose)

        # Camera motion speed (from GT)
        gt_pos = frame["pose"][:3, 3].numpy()
        motion_mm = float(np.linalg.norm(gt_pos - prev_gt_pos) * 1000)
        prev_gt_pos = gt_pos

        # Run tracking + mapping
        info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)

        # Alpha coverage: render from estimated pose, measure coverage
        with torch.no_grad():
            est_pose = slam.estimated_poses[-1]
            viewmat = torch.inverse(est_pose)
            rendered = renderer(gaussian_map, viewmat, K, W, H, downscale=4)
            alpha = rendered["alpha"].squeeze(-1)  # (H/4, W/4)
            alpha_coverage = float((alpha > 0.5).float().mean().item()) * 100

            # Depth validity: where GT depth > 0 AND alpha > 0.5
            gt_depth_ds = torch.nn.functional.interpolate(
                frame["depth"].to(device).unsqueeze(0), scale_factor=0.25, mode='nearest'
            ).squeeze(0).squeeze(0)
            depth_valid_mask = (gt_depth_ds > 0) & (alpha > 0.5)
            gt_valid = (gt_depth_ds > 0).float().sum()
            depth_validity = float(depth_valid_mask.float().sum() / gt_valid.clamp(min=1)) * 100

        ate_cm = info["ate"] * 100
        tracking_loss = info["tracking_loss"]
        n_gaussians = info["total_gaussians"]
        is_kf = info["is_keyframe"]

        row = {
            "frame": i,
            "ate_cm": round(ate_cm, 3),
            "tracking_loss": round(float(tracking_loss), 6),
            "alpha_coverage_pct": round(alpha_coverage, 1),
            "depth_validity_pct": round(depth_validity, 1),
            "gt_motion_mm": round(motion_mm, 2),
            "total_gaussians": n_gaussians,
            "is_keyframe": is_kf,
        }
        per_frame.append(row)

        # Print every 10 frames or when ATE spikes
        if i % 10 == 0 or ate_cm > 5.0 or i <= 5:
            kf_str = "YES" if is_kf else ""
            print(f"  {i:>6} {ate_cm:>8.2f} {tracking_loss:>8.4f} {alpha_coverage:>6.1f}% "
                  f"{depth_validity:>7.1f}% {motion_mm:>9.1f} {n_gaussians:>10} {kf_str:>8}")

    # Summary statistics
    ates = [f["ate_cm"] for f in per_frame]
    ate_rmse = float(np.sqrt(np.mean(np.array(ates) ** 2)))

    # Find drift onset: first frame where ATE > 3cm
    drift_onset = None
    for f in per_frame:
        if f["ate_cm"] > 3.0:
            drift_onset = f["frame"]
            break

    # Find correlation: what happened just before drift
    pre_drift_info = None
    if drift_onset and drift_onset > 5:
        pre_drift = [f for f in per_frame if drift_onset - 10 <= f["frame"] <= drift_onset]
        pre_drift_info = {
            "drift_onset_frame": drift_onset,
            "avg_alpha_before": round(np.mean([f["alpha_coverage_pct"] for f in pre_drift]), 1),
            "avg_motion_before_mm": round(np.mean([f["gt_motion_mm"] for f in pre_drift]), 2),
            "min_depth_validity_before": round(min(f["depth_validity_pct"] for f in pre_drift), 1),
            "avg_loss_before": round(np.mean([f["tracking_loss"] for f in pre_drift]), 6),
        }

    summary = {
        "scene": scene,
        "frames": len(dataset),
        "ate_rmse_cm": round(ate_rmse, 3),
        "ate_max_cm": round(max(ates), 3),
        "drift_onset_frame": drift_onset,
        "pre_drift_analysis": pre_drift_info,
        "avg_alpha_coverage": round(np.mean([f["alpha_coverage_pct"] for f in per_frame]), 1),
        "avg_depth_validity": round(np.mean([f["depth_validity_pct"] for f in per_frame]), 1),
        "avg_gt_motion_mm": round(np.mean([f["gt_motion_mm"] for f in per_frame]), 2),
        "max_gt_motion_mm": round(max(f["gt_motion_mm"] for f in per_frame), 2),
        "final_gaussians": per_frame[-1]["total_gaussians"],
        "per_frame": per_frame,
    }

    print(f"\n  === DIAGNOSIS for {scene} ===")
    print(f"  ATE RMSE: {ate_rmse:.2f} cm")
    print(f"  Drift onset: frame {drift_onset}" if drift_onset else "  Drift onset: none (< 3cm)")
    print(f"  Avg alpha coverage: {summary['avg_alpha_coverage']:.1f}%")
    print(f"  Avg depth validity: {summary['avg_depth_validity']:.1f}%")
    print(f"  Avg GT motion: {summary['avg_gt_motion_mm']:.1f} mm/frame")
    print(f"  Max GT motion: {summary['max_gt_motion_mm']:.1f} mm/frame")
    if pre_drift_info:
        print(f"  -- Before drift (frames {drift_onset-10} to {drift_onset}):")
        print(f"     Alpha coverage: {pre_drift_info['avg_alpha_before']:.1f}%")
        print(f"     Depth validity: {pre_drift_info['min_depth_validity_before']:.1f}%")
        print(f"     Avg motion: {pre_drift_info['avg_motion_before_mm']:.1f} mm/frame")
        print(f"     Avg tracking loss: {pre_drift_info['avg_loss_before']:.6f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Diagnose DynLang-SLAM tracking drift")
    parser.add_argument("--scenes", nargs="+", default=["room0", "room1", "room2"],
                        help="Scenes to diagnose")
    parser.add_argument("--frames", type=int, default=200, help="Max frames per scene")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f" DynLang-SLAM Diagnostic — {len(args.scenes)} scenes, {args.frames} frames each")
    print(f"{'='*80}")

    all_results = {}
    for scene in args.scenes:
        print(f"\n{'='*60}")
        print(f" Diagnosing: {scene}")
        print(f"{'='*60}")
        result = diagnose_scene(scene, args.frames)
        all_results[scene] = result

    # Cross-scene comparison
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f" Cross-Scene Comparison")
        print(f"{'='*80}")
        print(f"{'Scene':<10} {'ATE(cm)':>8} {'Drift@':>7} {'Alpha%':>7} {'DepVal%':>8} "
              f"{'AvgMot':>7} {'MaxMot':>7} {'Gauss':>8}")
        print(f"{'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")
        for scene, r in all_results.items():
            if "error" in r:
                continue
            drift = str(r["drift_onset_frame"]) if r["drift_onset_frame"] else "none"
            print(f"{scene:<10} {r['ate_rmse_cm']:>8.2f} {drift:>7} {r['avg_alpha_coverage']:>6.1f}% "
                  f"{r['avg_depth_validity']:>7.1f}% {r['avg_gt_motion_mm']:>6.1f} "
                  f"{r['max_gt_motion_mm']:>6.1f} {r['final_gaussians']:>8}")

    # Save
    out_path = Path("results") / "diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
