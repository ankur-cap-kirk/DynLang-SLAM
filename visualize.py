"""Visualize SLAM rendering quality.

Renders frames from the Gaussian map and shows side-by-side comparisons
with ground truth RGB and depth.

Usage:
    python visualize.py                          # Default: room0, 100 frames
    python visualize.py --frames 50              # Process 50 frames
    python visualize.py --scene room1            # Different scene
    python visualize.py --gt-pose                # Use GT poses (best quality)
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.core.renderer import GaussianRenderer
from dynlang_slam.slam.pipeline import SLAMPipeline


def depth_to_colormap(depth, vmin=None, vmax=None):
    """Convert depth map to a colormap (turbo-like) for visualization."""
    d = depth.copy()
    if vmin is None:
        vmin = d[d > 0].min() if (d > 0).any() else 0
    if vmax is None:
        vmax = d.max()
    d = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Simple turbo-like colormap
    r = np.clip(1.0 - 2.0 * np.abs(d - 0.75), 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(d - 0.5), 0, 1)
    b = np.clip(1.0 - 2.0 * np.abs(d - 0.25), 0, 1)

    # Make invalid depth (0) black
    mask = depth <= 0
    r[mask] = 0
    g[mask] = 0
    b[mask] = 0

    return np.stack([r, g, b], axis=-1)


def save_comparison(gt_rgb, gt_depth, pred_rgb, pred_depth, pred_alpha, path, frame_id):
    """Save a side-by-side comparison image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  pip install Pillow for visualization")
        return

    H, W = gt_rgb.shape[:2]

    # Convert to numpy uint8
    gt_rgb_np = (gt_rgb * 255).astype(np.uint8)
    pred_rgb_np = (np.clip(pred_rgb, 0, 1) * 255).astype(np.uint8)

    # RGB difference (amplified)
    rgb_diff = np.abs(gt_rgb - pred_rgb)
    rgb_diff_np = (np.clip(rgb_diff * 5, 0, 1) * 255).astype(np.uint8)  # 5x amplified

    # Depth colormaps
    vmin = gt_depth[gt_depth > 0].min() if (gt_depth > 0).any() else 0
    vmax = gt_depth.max()
    gt_depth_color = (depth_to_colormap(gt_depth, vmin, vmax) * 255).astype(np.uint8)
    pred_depth_color = (depth_to_colormap(pred_depth, vmin, vmax) * 255).astype(np.uint8)

    # Alpha map
    alpha_np = (np.clip(pred_alpha, 0, 1) * 255).astype(np.uint8)
    alpha_rgb = np.stack([alpha_np, alpha_np, alpha_np], axis=-1)

    # Create 2x3 grid: [GT RGB, Pred RGB, RGB Diff] / [GT Depth, Pred Depth, Alpha]
    pad = 4
    grid_h = 2 * H + 3 * pad
    grid_w = 3 * W + 4 * pad
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark gray bg

    # Row 1
    y0 = pad
    canvas[y0:y0+H, pad:pad+W] = gt_rgb_np
    canvas[y0:y0+H, 2*pad+W:2*pad+2*W] = pred_rgb_np
    canvas[y0:y0+H, 3*pad+2*W:3*pad+3*W] = rgb_diff_np

    # Row 2
    y1 = 2*pad + H
    canvas[y1:y1+H, pad:pad+W] = gt_depth_color
    canvas[y1:y1+H, 2*pad+W:2*pad+2*W] = pred_depth_color
    canvas[y1:y1+H, 3*pad+2*W:3*pad+3*W] = alpha_rgb

    img = Image.fromarray(canvas)

    # Add labels
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    labels = [
        (pad + 5, pad + 2, "GT RGB"),
        (2*pad + W + 5, pad + 2, "Rendered RGB"),
        (3*pad + 2*W + 5, pad + 2, "Error (5x)"),
        (pad + 5, 2*pad + H + 2, "GT Depth"),
        (2*pad + W + 5, 2*pad + H + 2, "Rendered Depth"),
        (3*pad + 2*W + 5, 2*pad + H + 2, f"Alpha (frame {frame_id})"),
    ]
    for x, y, text in labels:
        draw.text((x, y), text, fill=(255, 255, 0), font=font)

    img.save(str(path))


def main():
    parser = argparse.ArgumentParser(description="Visualize DynLang-SLAM rendering")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--gt-pose", action="store_true", help="Use GT poses")
    parser.add_argument("--viz-every", type=int, default=20, help="Save visualization every N frames")
    args = parser.parse_args()

    cfg = load_config(args.config, [])
    if args.scene:
        cfg.dataset.scene = args.scene
    cfg.dataset.max_frames = args.frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Load dataset
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

    # Output dir
    out_dir = Path("results") / cfg.dataset.scene / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process first frame
    first_frame = dataset[0]
    slam.process_first_frame(gaussian_map, first_frame)
    print(f"Map initialized: {gaussian_map.num_gaussians} Gaussians")

    # Save first frame visualization
    with torch.no_grad():
        pose0 = first_frame["pose"].to(device)
        viewmat0 = torch.inverse(pose0)
        result0 = renderer(gaussian_map, viewmat0, K,
                           cfg.dataset.image_width, cfg.dataset.image_height, downscale=1)

    gt_rgb0 = first_frame["rgb"].permute(1, 2, 0).cpu().numpy()
    gt_depth0 = first_frame["depth"].squeeze(0).cpu().numpy()
    pred_rgb0 = result0["rgb"].detach().cpu().numpy()
    pred_depth0 = result0["depth"].squeeze(-1).detach().cpu().numpy()
    pred_alpha0 = result0["alpha"].squeeze(-1).detach().cpu().numpy()

    save_comparison(gt_rgb0, gt_depth0, pred_rgb0, pred_depth0, pred_alpha0,
                    out_dir / "frame_000000.png", 0)
    print(f"  Saved: {out_dir / 'frame_000000.png'}")

    # SLAM loop
    print(f"\nRunning SLAM on {len(dataset)} frames...")
    for i in range(1, len(dataset)):
        frame = dataset[i]
        info = slam.process_frame(gaussian_map, frame, use_gt_pose=args.gt_pose)

        if i % 10 == 0:
            print(f"  Frame {i:4d}/{len(dataset)-1} | ATE: {info['ate']*100:.2f}cm | Gaussians: {info['total_gaussians']}")

        # Save visualization
        if i % args.viz_every == 0 or i == len(dataset) - 1:
            with torch.no_grad():
                est_pose = slam.estimated_poses[-1]
                viewmat = torch.inverse(est_pose)
                result = renderer(gaussian_map, viewmat, K,
                                  cfg.dataset.image_width, cfg.dataset.image_height, downscale=1)

            gt_rgb = frame["rgb"].permute(1, 2, 0).cpu().numpy()
            gt_depth = frame["depth"].squeeze(0).cpu().numpy()
            pred_rgb = result["rgb"].detach().cpu().numpy()
            pred_depth = result["depth"].squeeze(-1).detach().cpu().numpy()
            pred_alpha = result["alpha"].squeeze(-1).detach().cpu().numpy()

            save_path = out_dir / f"frame_{i:06d}.png"
            save_comparison(gt_rgb, gt_depth, pred_rgb, pred_depth, pred_alpha,
                            save_path, i)
            print(f"  Saved: {save_path}")

    print(f"\nDone! Visualizations saved to: {out_dir}")
    print(f"Open the folder to see side-by-side comparisons.")


if __name__ == "__main__":
    main()
