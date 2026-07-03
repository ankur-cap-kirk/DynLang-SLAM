"""H4 map build-up teaser - the analogue of midterm Fig. teaser
(all_scenes_progress).

For each H4 clip, we run SLAM (dyn ON, lang OFF for speed) and at four
timesteps we render the Gaussian map *from the same fixed canonical
viewpoint* (default: the estimated pose at frame 0). The result is a
2-row grid:

    Row 0 : input RGB at the milestone frame
    Row 1 : canonical-pose render at that timestep

Holding the viewpoint fixed makes it visually obvious that progressively
more of the scene is being mapped as the camera explores. Black panels in
the early columns are correct: at that moment, no Gaussians yet exist
inside the canonical-pose frustum.

Outputs:
    results/h4/figures/teaser_{clip}.png

Usage:
    python scripts/h4_teaser_progress.py --clip data/H4/clip_a --frames 90
    python scripts/h4_teaser_progress.py --clips clip_a,clip_b,clip_c \
        --frames 90 --output-dir results/h4/figures
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
    save_grid,
    to_uint8_rgb,
    frame_rgb_to_uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=str, default=None,
                   help="single-clip mode; takes precedence over --clips.")
    p.add_argument("--clips", type=str, default="clip_a",
                   help="comma-separated list when running multi-clip.")
    p.add_argument("--data-root", type=str, default="data/H4")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--milestones", type=str, default="0.10,0.30,0.60,0.95",
                   help="fractions of --frames at which to snapshot the canonical render.")
    p.add_argument("--output-dir", type=str, default="results/h4/figures")
    p.add_argument("--enable-dynamic", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _resolve_milestones(spec: str, n_frames: int) -> list[int]:
    fracs = [float(x) for x in spec.split(",") if x.strip()]
    return sorted(set(min(n_frames - 1, max(1, int(round(f * (n_frames - 1)))))
                      for f in fracs))


def _make_teaser(
    clip_dir: Path,
    n_frames: int,
    milestone_idxs: list[int],
    out_dir: Path,
    enable_dynamic: bool,
    device: str,
) -> Path:
    print(f"\n[teaser] >>> {clip_dir.name} (n_frames={n_frames}, milestones={milestone_idxs}) <<<")
    setup = load_h4_setup(
        clip_dir=clip_dir, n_frames=n_frames,
        enable_dynamic=enable_dynamic,
        enable_language=False,
        device=device,
    )
    slam, gmap, dataset = setup.slam, setup.gaussian_map, setup.dataset
    H, W = setup.intrinsics["height"], setup.intrinsics["width"]

    f0 = dataset[0]
    slam.process_first_frame(gmap, f0)

    # Canonical viewpoint = frame-0 estimated pose (= identity placeholder for H4).
    canonical_w2c = torch.linalg.inv(slam.estimated_poses[0]).to(setup.device)

    snapshots: list[tuple[int, np.ndarray, np.ndarray]] = []
    n_to_run = min(n_frames, len(dataset)) - 1
    t0 = time.time()
    for i in range(1, n_to_run + 1):
        frame = dataset[i]
        info = slam.process_frame(gmap, frame, use_gt_pose=False)
        if i in milestone_idxs:
            out = render_at_pose(slam, gmap, canonical_w2c, W, H, render_lang=False)
            rgb_uint8 = frame_rgb_to_uint8(frame)
            render_uint8 = to_uint8_rgb(out["rgb"])
            snapshots.append((i, rgb_uint8, render_uint8))
            print(f"  [snap] f{i:3d} | G={info['total_gaussians']:5d} | "
                  f"render mean={render_uint8.mean():.1f}")
        elif i % 10 == 0 or i == n_to_run:
            print(f"  f{i:3d}/{n_to_run} | G={info['total_gaussians']:5d}")
    print(f"[teaser] SLAM total {time.time() - t0:.1f}s")

    if not snapshots:
        raise RuntimeError("no snapshots collected; raise --frames or pick smaller --milestones.")

    snapshots.sort(key=lambda x: x[0])
    rgb_row = [s[1] for s in snapshots]
    render_row = [s[2] for s in snapshots]
    titles = [f"f{idx}" for (idx, _, _) in snapshots]

    out_png = out_dir / f"teaser_{clip_dir.name}.png"
    save_grid([rgb_row, render_row], out_png,
              titles_top=titles,
              row_labels=["Input RGB", "Canonical render"])
    print(f"[teaser] -> {out_png}")

    del slam, gmap, dataset, setup
    torch.cuda.empty_cache()
    return out_png


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    if args.clip is not None:
        clip_paths = [Path(args.clip) if Path(args.clip).is_absolute()
                      else (PROJECT_ROOT / args.clip)]
    else:
        clip_paths = [(data_root / c.strip()).resolve()
                      for c in args.clips.split(",") if c.strip()]

    milestone_idxs = _resolve_milestones(args.milestones, args.frames)

    for clip_dir in clip_paths:
        if not clip_dir.exists():
            print(f"[teaser] skipping missing {clip_dir}")
            continue
        try:
            _make_teaser(
                clip_dir=clip_dir.resolve(),
                n_frames=args.frames,
                milestone_idxs=milestone_idxs,
                out_dir=out_dir,
                enable_dynamic=args.enable_dynamic,
                device=args.device,
            )
        except Exception as e:
            print(f"[teaser] {clip_dir.name} failed: {e}")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
