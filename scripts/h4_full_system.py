"""H4 full-system table + per-clip filmstrip - the analogue of midterm
Tab. three_mode (full row) and Fig. full_system.

For each H4 clip we run a full DynLang-SLAM pass (dyn ON + lang ON), gather
the same set of system stats reported in the midterm full-system row
(drift, gaussian count, dyn fraction, language coverage, AE freeze frame,
total time + FPS), and produce a per-clip filmstrip figure with five
columns and four rows:

    Row 0 : input RGB
    Row 1 : SLAM render at the estimated pose
    Row 2 : YOLOv8 dynamic mask painted red
    Row 3 : 'person' relevancy heatmap

Outputs:
    results/h4/full_system/summary.{txt,csv,json}
    results/h4/full_system/{clip}_filmstrip.png
    results/h4/full_system/{clip}_perframe.csv

Usage:
    python scripts/h4_full_system.py
    python scripts/h4_full_system.py --clips clip_a --frames 90
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

from dynlang_slam.dynamic import DynamicDetector  # noqa: E402

from _h4_helpers import (  # noqa: E402
    load_h4_setup,
    render_at_pose,
    render_relevancy,
    apply_turbo,
    overlay_heatmap,
    paint_mask,
    save_grid,
    to_uint8_rgb,
    frame_rgb_to_uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/H4")
    p.add_argument("--clips", type=str, default="clip_a,clip_b,clip_c")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--n-panels", type=int, default=5)
    p.add_argument("--query", type=str, default="person")
    p.add_argument("--output", type=str, default="results/h4/full_system")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _drift_m(pose: torch.Tensor) -> float:
    return float(pose[:3, 3].norm().item())


def _resolve_panel_frames(n: int, total: int) -> list[int]:
    n = max(2, n)
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def _run_one_clip(
    clip_dir: Path,
    n_frames: int,
    n_panels: int,
    query: str,
    out_dir: Path,
    device: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"[full] >>> {clip_dir.name} ({n_frames} frames, dyn+lang ON) <<<")
    print(f"{'='*60}")
    setup = load_h4_setup(
        clip_dir=clip_dir, n_frames=n_frames,
        enable_dynamic=True, enable_language=True, device=device,
    )
    slam, gmap, dataset = setup.slam, setup.gaussian_map, setup.dataset
    H, W = setup.intrinsics["height"], setup.intrinsics["width"]

    panel_frame_idxs = _resolve_panel_frames(n_panels, min(n_frames, len(dataset)))

    f0 = dataset[0]
    slam.process_first_frame(gmap, f0)

    panel_data: dict[int, dict] = {}    # idx -> dict(rgb, render, dyn_pct, ...)
    csv_rows: list[dict] = []

    drifts: list[float] = []
    dyn_pcts: list[float] = []
    times: list[float] = []
    n_to_run = min(n_frames, len(dataset)) - 1
    ae_freeze_frame = -1
    t_start = time.time()
    for i in range(1, n_to_run + 1):
        frame = dataset[i]
        t0 = time.time()
        info = slam.process_frame(gmap, frame, use_gt_pose=False)
        dt = time.time() - t0
        times.append(dt)
        drifts.append(_drift_m(slam.estimated_poses[-1]))
        if "dynamic_pct" in info:
            dyn_pcts.append(float(info["dynamic_pct"]))
        ae_frozen = (slam._autoencoder.is_frozen
                     if slam._autoencoder is not None else False)
        if ae_frozen and ae_freeze_frame < 0:
            ae_freeze_frame = i

        csv_rows.append({
            "frame": i,
            "drift_m": drifts[-1],
            "dyn_pct": float(info.get("dynamic_pct", 0.0)),
            "n_gaussians": int(info["total_gaussians"]),
            "ae_frozen": int(ae_frozen),
            "frame_time_s": dt,
        })

        if i in panel_frame_idxs:
            est_pose_w2c = torch.linalg.inv(slam.estimated_poses[-1]).to(setup.device)
            out_render = render_at_pose(slam, gmap, est_pose_w2c, W, H, render_lang=False)
            panel_data[i] = {
                "rgb": frame_rgb_to_uint8(frame),
                "render": to_uint8_rgb(out_render["rgb"]),
                "dyn_pct": float(info.get("dynamic_pct", 0.0)),
                "ae_frozen": ae_frozen,
                "pose_w2c": est_pose_w2c.cpu(),
            }

        if i % 10 == 0 or i == n_to_run:
            print(f"  [{clip_dir.name}] frame {i:3d}/{n_to_run} | "
                  f"drift={drifts[-1]:5.2f} m | "
                  f"G={info['total_gaussians']:5d} | "
                  f"dyn={info.get('dynamic_pct', 0):.1f}% | "
                  f"AE_frozen={ae_frozen} | dt={dt:.2f} s")
    total_time = time.time() - t_start

    # Persist per-frame CSV.
    csv_path = out_dir / f"{clip_dir.name}_perframe.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        w.writeheader()
        w.writerows(csv_rows)

    # Per-Gaussian language coverage.
    lang_feats = gmap.lang_feats.data
    lang_norm = lang_feats.norm(dim=-1)
    lang_nonzero_pct = float((lang_norm > 0.01).float().mean().item()) * 100
    print(f"[full] AE freeze frame: {ae_freeze_frame}, "
          f"language coverage: {lang_nonzero_pct:.1f}%")

    # Detector for the dyn-mask row of the filmstrip.
    detector = DynamicDetector(
        model_name="yolov8x-seg",
        confidence_thresh=0.5,
        device=device,
        dynamic_classes=[0, 1, 2, 3, 5, 7, 14, 15, 16, 17],
    )

    # Build the filmstrip. Use percentile stretch across panels for relevancy
    # consistency.
    rels_per_panel: dict[int, np.ndarray] = {}
    if (slam._lang_initialized and slam._autoencoder is not None
            and slam._autoencoder.is_frozen):
        for idx in panel_frame_idxs:
            if idx not in panel_data:
                continue
            rel_dict = render_relevancy(
                slam, gmap, panel_data[idx]["pose_w2c"].to(setup.device),
                W, H, queries=[query],
            )
            rels_per_panel[idx] = rel_dict[query]
    else:
        print("[full] AE never froze; relevancy row will be all-grey.")

    if rels_per_panel:
        all_p5 = float(np.percentile(np.stack(list(rels_per_panel.values())), 5))
        all_p99 = float(np.percentile(np.stack(list(rels_per_panel.values())), 99))
    else:
        all_p5, all_p99 = 0.0, 1.0

    rgb_row: list[np.ndarray] = []
    render_row: list[np.ndarray] = []
    dyn_row: list[np.ndarray] = []
    rel_row: list[np.ndarray] = []
    titles: list[str] = []
    for idx in panel_frame_idxs:
        if idx not in panel_data:
            continue
        d = panel_data[idx]
        rgb_row.append(d["rgb"])
        render_row.append(d["render"])
        raw_dyn = detector.detect_and_merge(d["rgb"]).cpu().numpy().astype(bool)
        dyn_row.append(paint_mask(d["rgb"], raw_dyn, color=(255, 0, 0), alpha=0.55))
        if idx in rels_per_panel:
            heat = apply_turbo(rels_per_panel[idx], vmin=all_p5, vmax=all_p99)
            rel_row.append(overlay_heatmap(d["rgb"], heat, alpha=0.55))
        else:
            rel_row.append(d["rgb"])  # placeholder
        titles.append(f"f{idx} (drift={drifts[idx-1]:.2f} m)" if idx >= 1 else f"f{idx}")

    out_png = out_dir / f"{clip_dir.name}_filmstrip.png"
    save_grid(
        [rgb_row, render_row, dyn_row, rel_row],
        out_png,
        titles_top=titles,
        row_labels=["RGB", "Render", "Dyn mask", f"'{query}'"],
    )
    print(f"[full] -> {out_png}")

    summary = {
        "clip": clip_dir.name,
        "n_frames": n_to_run,
        "drift_end_m": drifts[-1] if drifts else 0.0,
        "drift_max_m": max(drifts) if drifts else 0.0,
        "n_gaussians": int(info["total_gaussians"]),
        "avg_dynamic_pct": float(np.mean(dyn_pcts)) if dyn_pcts else 0.0,
        "lang_nonzero_pct": lang_nonzero_pct,
        "ae_freeze_frame": ae_freeze_frame,
        "total_time_s": total_time,
        "fps": n_to_run / max(total_time, 1e-6),
    }

    del slam, gmap, dataset, setup
    torch.cuda.empty_cache()
    return summary


def _format_table(rows: list[dict]) -> str:
    out = []
    out.append(
        f"  {'Clip':<8s} {'Frames':>6s} {'Drift_m':>8s} {'#G':>8s} "
        f"{'Dyn%':>6s} {'Lang%':>7s} {'AE@':>5s} {'Time_s':>8s} {'FPS':>6s}"
    )
    out.append("  " + "-" * 75)
    for r in rows:
        out.append(
            f"  {r['clip']:<8s} {r['n_frames']:>6d} "
            f"{r['drift_end_m']:>8.3f} {r['n_gaussians']:>8d} "
            f"{r['avg_dynamic_pct']:>6.1f} {r['lang_nonzero_pct']:>7.1f} "
            f"{r['ae_freeze_frame']:>5d} "
            f"{r['total_time_s']:>8.1f} {r['fps']:>6.2f}"
        )
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    clip_names = [c.strip() for c in args.clips.split(",") if c.strip()]

    rows: list[dict] = []
    for cname in clip_names:
        clip_dir = (data_root / cname).resolve()
        if not clip_dir.exists():
            print(f"[full] skipping missing {clip_dir}")
            continue
        try:
            summary = _run_one_clip(
                clip_dir=clip_dir,
                n_frames=args.frames,
                n_panels=args.n_panels,
                query=args.query,
                out_dir=out_dir,
                device=args.device,
            )
            rows.append(summary)
        except Exception as e:
            print(f"[full] {cname} failed: {e}")
            import traceback
            traceback.print_exc()

    table = _format_table(rows)
    print("\n" + "=" * 60)
    print("H4 full-system summary (dynamic + language ON)")
    print("=" * 60)
    print(table)

    (out_dir / "summary.txt").write_text(table + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\nResults written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
