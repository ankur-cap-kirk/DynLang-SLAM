"""H4 Bayesian dynamic-belief evolution figure - the analogue of midterm
Fig. belief.

Runs SLAM with dynamic masking ON (and language OFF for speed) on one H4
clip, then rasterizes the per-Gaussian belief field through gsplat at
several timesteps and overlays it on the corresponding RGB.

Output (data/H4 paths abbreviated):
    results/h4/figures/belief_evolution_{clip}.png
        2-row grid: top = input RGB, bottom = belief overlay (turbo)
    results/h4/figures/belief_evolution_{clip}.csv
        per-frame max/mean belief, alpha coverage of the belief mass.

Usage:
    python scripts/h4_belief_evolution.py --clip data/H4/clip_a --frames 90
    python scripts/h4_belief_evolution.py --clip data/H4/clip_a --snapshots 5,30,60,90
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _h4_helpers import (  # noqa: E402
    load_h4_setup,
    apply_turbo,
    overlay_heatmap,
    save_grid,
    frame_rgb_to_uint8,
    render_per_gaussian_scalar,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=str, default="data/H4/clip_a")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--snapshots", type=str, default="",
                   help='comma-separated frame indices, e.g. "5,30,60,90". '
                        'If empty, picks 4 evenly-spaced points.')
    p.add_argument("--output-dir", type=str, default="results/h4/figures")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _resolve_snapshots(spec: str, n_frames: int) -> list[int]:
    if spec.strip():
        return sorted(set(int(x) for x in spec.split(",") if x.strip()))
    # 4 panels, evenly spaced (skip frame 0 which is always empty).
    return [int(round(p * (n_frames - 1))) for p in (0.05, 0.30, 0.60, 0.95)]


def _maybe_clip_dir(arg: str) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def main() -> int:
    args = parse_args()
    clip_dir = _maybe_clip_dir(args.clip)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = set(_resolve_snapshots(args.snapshots, args.frames))
    print(f"[belief] clip={clip_dir.name}  frames={args.frames}  snapshots={sorted(snapshots)}")

    setup = load_h4_setup(
        clip_dir=clip_dir, n_frames=args.frames,
        enable_dynamic=True,
        enable_language=False,
        device=args.device,
    )
    slam, gmap, dataset = setup.slam, setup.gaussian_map, setup.dataset
    H, W = setup.intrinsics["height"], setup.intrinsics["width"]

    f0 = dataset[0]
    slam.process_first_frame(gmap, f0)

    snapshot_panels: list[tuple[int, np.ndarray, np.ndarray, dict]] = []
    csv_rows: list[dict] = []
    n_to_run = min(args.frames, len(dataset)) - 1
    t0_total = time.time()
    for i in range(1, n_to_run + 1):
        frame = dataset[i]
        info = slam.process_frame(gmap, frame, use_gt_pose=False)
        belief = gmap.dynamic_belief.detach()
        max_b = float(belief.max().item()) if belief.numel() else 0.0
        mean_b = float(belief.mean().item()) if belief.numel() else 0.0
        n_hot = int((belief > 0.05).sum().item())
        csv_rows.append({
            "frame": i,
            "max_belief": max_b,
            "mean_belief": mean_b,
            "hot_gaussians": n_hot,
            "n_gaussians": int(info["total_gaussians"]),
            "dynamic_pct": float(info.get("dynamic_pct", 0.0)),
        })

        if i in snapshots:
            est_pose_w2c = torch.linalg.inv(slam.estimated_poses[-1]).to(setup.device)
            scalar_img, alpha_img = render_per_gaussian_scalar(
                gmap, belief, est_pose_w2c, slam.K, W, H,
            )
            rgb_uint8 = frame_rgb_to_uint8(frame)
            snapshot_panels.append((i, rgb_uint8, scalar_img, {
                "max_belief": max_b, "mean_belief": mean_b, "n_hot": n_hot,
            }))
            print(f"  [snap] frame {i:3d} | max_b={max_b:.3f} mean_b={mean_b:.4f} hot={n_hot}")
        elif i % 10 == 0 or i == n_to_run:
            print(f"  frame {i:3d}/{n_to_run} | dyn={info.get('dynamic_pct', 0):.1f}% | "
                  f"max_b={max_b:.3f} | G={info['total_gaussians']}")

    total_time = time.time() - t0_total
    print(f"[belief] SLAM total time: {total_time:.1f} s")

    if not snapshot_panels:
        print("[belief] no snapshots fell within the run; nothing to plot.")
        return 1

    # Compose the grid: top row RGB, bottom row belief overlay.
    snapshot_panels.sort(key=lambda x: x[0])
    rgb_row = [s[1] for s in snapshot_panels]
    overlay_row: list[np.ndarray] = []
    for (idx, rgb_uint8, scalar_img, meta) in snapshot_panels:
        # Stretch to a common scale across snapshots so panels are comparable.
        all_max = max((s[2].max() for s in snapshot_panels), default=1.0)
        all_max = max(float(all_max), 1e-6)
        heat = apply_turbo(scalar_img, vmin=0.0, vmax=all_max)
        overlay = overlay_heatmap(rgb_uint8, heat, alpha=0.55)
        overlay_row.append(overlay)

    titles = [f"f{idx} (max_b={meta['max_belief']:.2f})"
              for (idx, _, _, meta) in snapshot_panels]
    out_png = out_dir / f"belief_evolution_{clip_dir.name}.png"
    save_grid([rgb_row, overlay_row], out_png,
              titles_top=titles,
              row_labels=["RGB", "Belief b in [0,1]"])
    print(f"[belief] -> {out_png}")

    csv_path = out_dir / f"belief_evolution_{clip_dir.name}.csv"
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"[belief] -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
