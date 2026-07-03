"""Minimal end-to-end sanity check for the H4 (in-the-wild Pixel) pipeline.

What this checks (in order):
  1. data/H4/<clip>/ has rgb.txt, depth.txt, intrinsics.txt and the PNGs.
  2. TUMDataset can load the sequence with has_gt=False.
  3. GaussianMap initialises from the first frame (depth -> 3D points).
  4. SLAMPipeline runs N frames without crashing.
  5. Tracking stays bounded - drift from the first-frame pose is reported.

We DELIBERATELY disable dynamic masking and language features here. The goal
is to isolate "does monocular-depth + tracker + mapper wire together at all"
from the harder downstream questions (does dynamic masking improve quality,
does language warm-up converge, etc.). Those are tested by separate scripts.

If this script prints `[smoke] PASSED` you can move on to the per-figure
scripts. If it crashes or drift looks unreasonable (>5 m over 50 frames of
walking), we have a problem to debug before running anything else.

Usage:
    python scripts/h4_smoke_test.py --clip data/H4/clip_a
    python scripts/h4_smoke_test.py --clip data/H4/clip_a --frames 30 --device cpu
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

from dynlang_slam.utils.config import load_config  # noqa: E402
from dynlang_slam.data.tum import (  # noqa: E402
    TUMDataset,
    load_intrinsics_from_file,
    get_pixel8_portrait_intrinsics,
)
from dynlang_slam.core.gaussians import GaussianMap  # noqa: E402
from dynlang_slam.slam.pipeline import SLAMPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=str, default="data/H4/clip_a",
                   help="Path to a clip directory written by prepare_h4.py.")
    p.add_argument("--config", type=str, default="configs/h4.yaml")
    p.add_argument("--frames", type=int, default=50,
                   help="Frames to process beyond the init frame (default 50).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-drift-m", type=float, default=5.0,
                   help="Hard fail if drift from origin exceeds this (default 5 m).")
    return p.parse_args()


def _check_layout(clip_dir: Path) -> None:
    required = ["rgb.txt", "depth.txt", "intrinsics.txt", "rgb", "depth"]
    missing = [name for name in required if not (clip_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"[smoke] FAIL: clip dir {clip_dir} is missing: {missing}.\n"
            "Run scripts/prepare_h4.py first."
        )
    n_rgb = len(list((clip_dir / "rgb").glob("*.png")))
    n_depth = len(list((clip_dir / "depth").glob("*.png")))
    print(f"[smoke] layout OK: {n_rgb} RGB / {n_depth} depth in {clip_dir}")
    if n_rgb == 0 or n_depth == 0:
        raise SystemExit("[smoke] FAIL: empty rgb/ or depth/ directory.")
    if abs(n_rgb - n_depth) > 1:
        print(f"[smoke] WARN: |rgb - depth| = {abs(n_rgb - n_depth)} > 1; "
              "intrinsics association may drop frames.")


def _drift_from_origin(pose: torch.Tensor) -> float:
    return float(pose[:3, 3].norm().item())


def main() -> int:
    args = parse_args()
    clip_dir = Path(args.clip).resolve()
    print(f"[smoke] clip = {clip_dir}")
    _check_layout(clip_dir)

    cfg = load_config(args.config, [])
    cfg.dynamic.enabled = False
    cfg.language.enabled = False
    cfg.dataset.has_gt = False
    cfg.dataset.max_frames = args.frames + 1

    h, w = cfg.dataset.image_height, cfg.dataset.image_width

    intrinsics_file = clip_dir / "intrinsics.txt"
    if intrinsics_file.exists():
        intrinsics = load_intrinsics_from_file(str(intrinsics_file), h, w)
        print(f"[smoke] using intrinsics from {intrinsics_file}: "
              f"fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
              f"cx={intrinsics['cx']:.1f} cy={intrinsics['cy']:.1f}")
    else:
        intrinsics = get_pixel8_portrait_intrinsics(h, w)
        print("[smoke] WARN: intrinsics.txt missing, falling back to defaults.")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[smoke] CUDA unavailable, falling back to CPU.")
        device = "cpu"

    dataset = TUMDataset(
        data_dir=str(clip_dir),
        height=h,
        width=w,
        depth_scale=cfg.dataset.depth_scale,
        max_frames=cfg.dataset.max_frames,
        has_gt=False,
    )
    print(f"[smoke] dataset frames after association: {len(dataset)}")
    if len(dataset) < 5:
        raise SystemExit("[smoke] FAIL: <5 frames, cannot run a meaningful test.")

    f0 = dataset[0]
    print(f"[smoke] frame[0] rgb dtype={f0['rgb'].dtype} shape={tuple(f0['rgb'].shape)}")
    print(f"[smoke] frame[0] depth dtype={f0['depth'].dtype} shape={tuple(f0['depth'].shape)} "
          f"min={float(f0['depth'].min()):.3f} m  "
          f"med={float(f0['depth'].median()):.3f} m  "
          f"max={float(f0['depth'].max()):.3f} m")
    if float(f0['depth'].max()) <= 0:
        raise SystemExit("[smoke] FAIL: depth is all zero. Did prepare_h4 run with --skip-depth?")
    if float(f0['depth'].max()) > 200:
        print("[smoke] WARN: depth max > 200 m; outdoor depth may be unclipped.")

    gaussian_map = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)

    slam.process_first_frame(gaussian_map, f0)
    print(f"[smoke] init complete: {gaussian_map.num_gaussians} gaussians")
    if device == "cuda":
        print(f"[smoke] VRAM after init: "
              f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    drifts: list[float] = []
    times: list[float] = []
    n_to_run = min(args.frames, len(dataset) - 1)
    print(f"[smoke] running {n_to_run} frames...")
    for i in range(1, n_to_run + 1):
        frame = dataset[i]
        t0 = time.time()
        info = slam.process_frame(gaussian_map, frame, use_gt_pose=False)
        dt = time.time() - t0
        drift = _drift_from_origin(slam.estimated_poses[-1])
        drifts.append(drift)
        times.append(dt)
        if i % 5 == 0 or i == n_to_run:
            kf = "KF" if info.get("is_keyframe") else "  "
            print(f"  [{kf}] frame {i:3d} | drift={drift:5.2f} m | "
                  f"G={info['total_gaussians']:6d} | dt={dt:.2f} s")
        if drift > args.max_drift_m:
            print(f"[smoke] FAIL: drift {drift:.2f} m exceeds --max-drift-m "
                  f"({args.max_drift_m:.2f} m).")
            return 1

    avg_dt = float(np.mean(times)) if times else 0.0
    final_drift = drifts[-1] if drifts else 0.0
    print()
    print(f"[smoke] frames processed   : {n_to_run}")
    print(f"[smoke] final gaussians    : {gaussian_map.num_gaussians}")
    print(f"[smoke] final drift (m)    : {final_drift:.3f}")
    print(f"[smoke] mean per-frame (s) : {avg_dt:.2f}  ({1.0 / max(avg_dt, 1e-6):.2f} fps)")
    if device == "cuda":
        print(f"[smoke] peak VRAM (GB)     : "
              f"{torch.cuda.max_memory_allocated() / (1024**3):.2f}")
    print()
    print("[smoke] PASSED")
    print("[smoke] Next: run the dynamic-masking ablation:")
    print(f"        python scripts/h4_dyn_ablation.py --clip {clip_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
