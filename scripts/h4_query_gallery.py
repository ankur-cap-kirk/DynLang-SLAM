"""H4 open-vocabulary query gallery - the analogue of midterm Fig. gallery.

Runs the full DynLang-SLAM pipeline (dynamic + language ON) on one H4 clip
for long enough that the autoencoder warmup completes and language features
are non-zero on most Gaussians, then renders relevancy heatmaps for a small
set of outdoor-friendly text queries.

The script saves both the figure (2-row grid: RGB | overlayed query) and a
.npz containing the rendered language feature map at the chosen viewpoint
(so h4_iou_query.py can reuse it without re-running SLAM).

Usage:
    python scripts/h4_query_gallery.py --clip data/H4/clip_a --frames 90
    python scripts/h4_query_gallery.py --clip data/H4/clip_a \
        --queries "person,tree,bench,car,sidewalk,building"
"""

from __future__ import annotations

import argparse
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
    render_at_pose,
    render_relevancy,
    apply_turbo,
    overlay_heatmap,
    save_grid,
    to_uint8_rgb,
    frame_rgb_to_uint8,
)


DEFAULT_QUERIES = "person,tree,bench,car,sidewalk,building"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=str, default="data/H4/clip_a")
    p.add_argument("--frames", type=int, default=90,
                   help="frames of SLAM. Need >= ~50 for AE warmup to freeze.")
    p.add_argument("--query-frame", type=int, default=-1,
                   help="-1 = use the last frame's pose for the query view.")
    p.add_argument("--queries", type=str, default=DEFAULT_QUERIES)
    p.add_argument("--output-dir", type=str, default="results/h4/figures")
    p.add_argument("--cache-dir", type=str, default="results/h4/cache",
                   help="where the rendered language feature map is saved for reuse.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    clip_dir = Path(args.clip)
    if not clip_dir.is_absolute():
        clip_dir = PROJECT_ROOT / clip_dir
    clip_dir = clip_dir.resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    if len(queries) < 2:
        raise SystemExit("need at least 2 queries to compose a gallery.")

    setup = load_h4_setup(
        clip_dir=clip_dir,
        n_frames=args.frames,
        enable_dynamic=True,
        enable_language=True,
        device=args.device,
    )
    slam, gmap, dataset = setup.slam, setup.gaussian_map, setup.dataset
    H, W = setup.intrinsics["height"], setup.intrinsics["width"]

    print(f"[gallery] running SLAM on {clip_dir.name} for {args.frames} frames"
          f" (lang+dyn ON)...")
    t_total = time.time()
    f0 = dataset[0]
    slam.process_first_frame(gmap, f0)
    n_to_run = min(args.frames, len(dataset)) - 1
    for i in range(1, n_to_run + 1):
        frame = dataset[i]
        info = slam.process_frame(gmap, frame, use_gt_pose=False)
        if i % 10 == 0 or i == n_to_run:
            ae_frozen = (slam._autoencoder.is_frozen
                         if slam._autoencoder is not None else False)
            print(f"  frame {i:3d}/{n_to_run} | G={info['total_gaussians']:5d} | "
                  f"dyn={info.get('dynamic_pct', 0):.1f}% | AE_frozen={ae_frozen}")
    total_time = time.time() - t_total
    print(f"[gallery] SLAM total {total_time:.1f}s")

    if not (slam._lang_initialized and slam._autoencoder is not None
            and slam._autoencoder.is_frozen):
        print("[gallery] WARNING: AE never froze in this run.")
        print("          Language features are at zero-init; relevancy will be ~0.5 everywhere.")
        print("          Try a longer --frames value, or lower the AE warmup threshold.")

    # Choose the query viewpoint.
    qf = args.query_frame
    if qf < 0 or qf >= len(slam.estimated_poses):
        qf = len(slam.estimated_poses) - 1
    print(f"[gallery] query viewpoint = estimated pose at frame {qf}")
    pose_w2c = torch.linalg.inv(slam.estimated_poses[qf]).to(setup.device)

    # Render input RGB at that frame for the top row of the gallery.
    query_frame = dataset[qf]
    rgb_uint8 = frame_rgb_to_uint8(query_frame)

    # Render lang feature map once + save for IoU reuse.
    out = render_at_pose(slam, gmap, pose_w2c, W, H, render_lang=True)
    if "lang_feats" not in out:
        raise SystemExit("language render failed; gaussian_map.lang_feats missing?")
    lang_feat_map = out["lang_feats"].detach().cpu().numpy()  # (H, W, D)

    cache_path = cache_dir / f"{clip_dir.name}_qf{qf}_langfeat.npz"
    np.savez(cache_path,
             lang_feat=lang_feat_map,
             rgb=rgb_uint8,
             query_frame_idx=qf,
             height=H, width=W)
    print(f"[gallery] cached language feature map -> {cache_path}")

    # Run the queries (relevancy uses the same render, just different text vectors).
    rels = render_relevancy(slam, gmap, pose_w2c, W, H, queries=queries)

    # Compose grid: top row = always RGB (so it's clear what the viewpoint shows),
    # bottom row = per-query relevancy overlay
    top_row = [rgb_uint8.copy() for _ in queries]
    bot_row: list[np.ndarray] = []
    for q in queries:
        rel = rels[q]
        # Percentile stretch makes weakly differentiated heatmaps readable.
        vmin = float(np.percentile(rel, 5))
        vmax = float(np.percentile(rel, 99))
        heat = apply_turbo(rel, vmin=vmin, vmax=vmax)
        bot_row.append(overlay_heatmap(rgb_uint8, heat, alpha=0.6))
        print(f"  '{q:<10s}': mean={rel.mean():.3f} p5={vmin:.3f} p99={vmax:.3f}")

    out_png = out_dir / f"query_gallery_{clip_dir.name}_qf{qf}.png"
    save_grid([top_row, bot_row], out_png,
              titles_top=queries,
              row_labels=["RGB", "Relevancy"])
    print(f"[gallery] -> {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
