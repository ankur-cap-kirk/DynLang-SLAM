"""H4 dynamic-masking ablation table - the analogue of midterm Tab. dyn_ablation
and Tab. three_mode (drift columns) and the canonical-pose render PSNR column.

For each H4 clip we run two full SLAM passes - identical except for
``cfg.dynamic.enabled`` - and report:

    drift_end_m       : ||T_N - T_0||  (no GT; use this as the proxy for
                                        tracker accuracy. More drift on
                                        dyn_off means the moving people
                                        polluted the tracker).
    canonical_psnr_dB : PSNR of the canonical-pose render at frame N
                        against the *first-frame RGB* (a measure of map
                        cleanliness; the static parts of the scene
                        should look like the first frame's static parts).
    n_gaussians       : final Gaussian count.
    avg_dynamic_pct   : average per-frame dynamic-area fraction
                        (only meaningful in dyn_on runs).
    total_time_s      : wall clock.

We also save per-frame drift CSVs so the user can plot them later.

Usage:
    python scripts/h4_dyn_ablation.py
    python scripts/h4_dyn_ablation.py --clips clip_a,clip_b --frames 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _h4_helpers import load_h4_setup, render_at_pose, to_uint8_rgb, frame_rgb_to_uint8  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/H4")
    p.add_argument("--clips", type=str, default="clip_a,clip_b,clip_c")
    p.add_argument("--frames", type=int, default=60,
                   help="frames per run (default 60: enough for several KFs and a meaningful drift signal).")
    p.add_argument("--output", type=str, default="results/h4/dyn_ablation")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _drift_m(pose: torch.Tensor) -> float:
    return float(pose[:3, 3].norm().item())


def _psnr(a_uint8: np.ndarray, b_uint8: np.ndarray) -> float:
    """PSNR between two HxWx3 uint8 images (using only pixels where both are non-black)."""
    if a_uint8.shape != b_uint8.shape:
        h = min(a_uint8.shape[0], b_uint8.shape[0])
        w = min(a_uint8.shape[1], b_uint8.shape[1])
        a_uint8, b_uint8 = a_uint8[:h, :w], b_uint8[:h, :w]
    a = a_uint8.astype(np.float32)
    b = b_uint8.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-6:
        return 99.0
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def _run_one(
    clip_dir: Path,
    n_frames: int,
    enable_dynamic: bool,
    device: str,
    out_dir: Path,
) -> dict:
    tag = "dyn_on" if enable_dynamic else "dyn_off"
    print(f"\n[ablation] >>> {clip_dir.name} / {tag} ({n_frames} frames) <<<")
    setup = load_h4_setup(
        clip_dir=clip_dir, n_frames=n_frames,
        enable_dynamic=enable_dynamic,
        enable_language=False,         # ablation only
        device=device,
    )
    slam, gmap, dataset = setup.slam, setup.gaussian_map, setup.dataset

    f0 = dataset[0]
    slam.process_first_frame(gmap, f0)

    drifts: list[float] = []
    dyn_pcts: list[float] = []
    times: list[float] = []
    n_to_run = min(n_frames, len(dataset)) - 1
    t_total0 = time.time()
    for i in range(1, n_to_run + 1):
        frame = dataset[i]
        t0 = time.time()
        info = slam.process_frame(gmap, frame, use_gt_pose=False)
        dt = time.time() - t0
        times.append(dt)
        drifts.append(_drift_m(slam.estimated_poses[-1]))
        if "dynamic_pct" in info:
            dyn_pcts.append(float(info["dynamic_pct"]))
        if i % 10 == 0 or i == n_to_run:
            print(f"  [{tag}] frame {i:3d}/{n_to_run} | drift={drifts[-1]:5.2f} m | "
                  f"G={info['total_gaussians']:6d} | dt={dt:.2f} s")
    total_time = time.time() - t_total0

    # Canonical-pose render (= frame 0 estimated viewmat).
    f0_pose_w2c = torch.linalg.inv(slam.estimated_poses[0]).to(setup.device)
    out = render_at_pose(slam, gmap, f0_pose_w2c,
                         width=setup.intrinsics["width"],
                         height=setup.intrinsics["height"], render_lang=False)
    canonical_render_uint8 = to_uint8_rgb(out["rgb"])
    f0_rgb_uint8 = frame_rgb_to_uint8(f0)
    canonical_psnr = _psnr(f0_rgb_uint8, canonical_render_uint8)

    # Persist per-frame CSV
    csv_path = out_dir / f"{clip_dir.name}_{tag}_perframe.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "drift_m", "dyn_pct", "frame_time_s"])
        for i, drift in enumerate(drifts, start=1):
            dp = dyn_pcts[i - 1] if i - 1 < len(dyn_pcts) else 0.0
            ft = times[i - 1] if i - 1 < len(times) else 0.0
            w.writerow([i, f"{drift:.6f}", f"{dp:.3f}", f"{ft:.3f}"])

    # Persist canonical render for visual sanity
    import cv2
    cv2.imwrite(
        str(out_dir / f"{clip_dir.name}_{tag}_canonical.png"),
        cv2.cvtColor(canonical_render_uint8, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    cv2.imwrite(
        str(out_dir / f"{clip_dir.name}_{tag}_first_rgb.png"),
        cv2.cvtColor(f0_rgb_uint8, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )

    # Cleanup before next run
    del slam, gmap, dataset, setup
    torch.cuda.empty_cache()

    return {
        "clip": clip_dir.name,
        "mode": tag,
        "n_frames": n_to_run,
        "drift_end_m": drifts[-1] if drifts else 0.0,
        "drift_max_m": max(drifts) if drifts else 0.0,
        "canonical_psnr_dB": canonical_psnr,
        "n_gaussians": int(info["total_gaussians"]),
        "avg_dynamic_pct": float(np.mean(dyn_pcts)) if dyn_pcts else 0.0,
        "total_time_s": total_time,
        "fps": n_to_run / max(total_time, 1e-6),
    }


def _format_table(rows: list[dict]) -> str:
    lines = []
    lines.append(
        f"  {'Clip':<8s} {'Mode':<8s} "
        f"{'Drift_end_m':>12s} {'Canon_PSNR_dB':>14s} "
        f"{'#G':>8s} {'Dyn%':>7s} {'Time_s':>8s} {'FPS':>6s}"
    )
    lines.append("  " + "-" * 80)
    for r in rows:
        lines.append(
            f"  {r['clip']:<8s} {r['mode']:<8s} "
            f"{r['drift_end_m']:>12.3f} {r['canonical_psnr_dB']:>14.2f} "
            f"{r['n_gaussians']:>8d} {r['avg_dynamic_pct']:>7.1f} "
            f"{r['total_time_s']:>8.1f} {r['fps']:>6.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_root = (PROJECT_ROOT / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    clip_names = [c.strip() for c in args.clips.split(",") if c.strip()]
    out_dir = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cname in clip_names:
        clip_dir = data_root / cname
        if not clip_dir.exists():
            print(f"[ablation] skipping missing {clip_dir}")
            continue
        for enable_dyn in (False, True):
            row = _run_one(clip_dir, args.frames, enable_dyn, args.device, out_dir)
            rows.append(row)

    table_text = _format_table(rows)
    print("\n" + "=" * 60)
    print("H4 dynamic-masking ablation summary")
    print("=" * 60)
    print(table_text)

    (out_dir / "summary.txt").write_text(table_text + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        if rows:
            w.writerow(list(rows[0].keys()))
            for r in rows:
                w.writerow([r[k] for k in rows[0].keys()])

    # Per-clip pairwise delta (psnr_dyn_on - psnr_dyn_off, drift_dyn_on - drift_dyn_off)
    deltas = []
    for cname in clip_names:
        on_row = next((r for r in rows if r["clip"] == cname and r["mode"] == "dyn_on"), None)
        off_row = next((r for r in rows if r["clip"] == cname and r["mode"] == "dyn_off"), None)
        if on_row and off_row:
            d = {
                "clip": cname,
                "delta_psnr_dB": on_row["canonical_psnr_dB"] - off_row["canonical_psnr_dB"],
                "delta_drift_m": on_row["drift_end_m"] - off_row["drift_end_m"],
                "delta_gaussians": on_row["n_gaussians"] - off_row["n_gaussians"],
            }
            deltas.append(d)
    if deltas:
        print("\n[ablation] dyn_on minus dyn_off:")
        for d in deltas:
            print(f"  {d['clip']}: dPSNR={d['delta_psnr_dB']:+.2f} dB | "
                  f"dDrift={d['delta_drift_m']:+.2f} m | "
                  f"dG={d['delta_gaussians']:+d}")
        with open(out_dir / "deltas.json", "w") as f:
            json.dump(deltas, f, indent=2)

    print(f"\nResults written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
