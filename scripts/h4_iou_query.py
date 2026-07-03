"""H4 quantitative IoU of the rendered "person" query against the YOLO mask -
the analogue of midterm Tab. iou_person and the IoU column.

Pipeline:
  1. Run full SLAM (dyn+lang ON) for N frames so the map and language
     features mature.
  2. For each evaluation frame (default 30%, 60%, 95% of the run):
       a. Render the language feature map at that frame's estimated pose.
       b. Compute per-pixel "person" relevancy using the same recipe as
          h4_query_gallery.py.
       c. Run YOLOv8 person-class segmentation on the input RGB to get the
          ground-truth silhouette.
       d. For top-k% in {5, 10, 20, 30}: threshold the relevancy at that
          intensity quantile and compute IoU / precision / recall against
          the YOLO mask.
  3. Plot IoU vs top-k% (band = std over the evaluated frames) and write
     an IoU/precision/recall table.

Outputs:
    results/h4/figures/iou_query_{clip}.png
    results/h4/figures/iou_query_{clip}.csv
    results/h4/figures/iou_query_{clip}.txt

Usage:
    python scripts/h4_iou_query.py --clip data/H4/clip_a --frames 90
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dynlang_slam.dynamic import DynamicDetector  # noqa: E402

from _h4_helpers import (  # noqa: E402
    load_h4_setup,
    render_at_pose,
    render_relevancy,
    frame_rgb_to_uint8,
)


DEFAULT_KS = (5, 10, 20, 30)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=str, default="data/H4/clip_a")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--query", type=str, default="person")
    p.add_argument("--eval-fracs", type=str, default="0.30,0.60,0.95")
    p.add_argument("--top-ks", type=str, default=",".join(str(k) for k in DEFAULT_KS))
    p.add_argument("--output-dir", type=str, default="results/h4/figures")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _resolve_eval_frames(spec: str, n_frames: int) -> list[int]:
    fracs = [float(x) for x in spec.split(",") if x.strip()]
    return sorted(set(min(n_frames - 1, max(1, int(round(f * (n_frames - 1)))))
                      for f in fracs))


def _iou_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """All inputs (H, W) bool. Returns (iou, precision, recall)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter) / float(union) if union > 0 else 0.0
    p = float(inter) / float(pred.sum()) if pred.sum() > 0 else 0.0
    r = float(inter) / float(gt.sum()) if gt.sum() > 0 else 0.0
    return iou, p, r


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
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]

    # 1. Full SLAM
    setup = load_h4_setup(
        clip_dir=clip_dir,
        n_frames=args.frames,
        enable_dynamic=True,
        enable_language=True,
        device=args.device,
    )
    slam, gmap, dataset = setup.slam, setup.gaussian_map, setup.dataset
    H, W = setup.intrinsics["height"], setup.intrinsics["width"]

    print(f"[iou] SLAM on {clip_dir.name} for {args.frames} frames...")
    t0 = time.time()
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
                  f"AE_frozen={ae_frozen}")
    print(f"[iou] SLAM total {time.time() - t0:.1f}s")

    if not (slam._lang_initialized and slam._autoencoder is not None
            and slam._autoencoder.is_frozen):
        print("[iou] WARNING: AE never froze. Numbers below are not meaningful.")

    # 2. Per-frame YOLO + per-frame relevancy
    eval_frames = _resolve_eval_frames(args.eval_fracs, args.frames)
    print(f"[iou] eval frames: {eval_frames}")

    detector = DynamicDetector(
        model_name="yolov8x-seg",
        confidence_thresh=0.5,
        device=args.device,
        # Person-only for the GT-mask (we restrict the query to a single class).
        dynamic_classes=[0],
    )

    rows: list[dict] = []
    per_frame_rels: list[np.ndarray] = []
    per_frame_gts: list[np.ndarray] = []
    for ef in eval_frames:
        if ef >= len(slam.estimated_poses):
            print(f"[iou] frame {ef} out of range (have {len(slam.estimated_poses)} poses); skipping.")
            continue
        pose_w2c = torch.linalg.inv(slam.estimated_poses[ef]).to(setup.device)
        rel_dict = render_relevancy(slam, gmap, pose_w2c, W, H, queries=[args.query])
        rel = rel_dict[args.query]                  # (H, W) in [0,1]
        rgb_uint8 = frame_rgb_to_uint8(dataset[ef])
        gt_mask = detector.detect_and_merge(rgb_uint8).cpu().numpy().astype(bool)

        per_frame_rels.append(rel)
        per_frame_gts.append(gt_mask)

        for k in top_ks:
            thresh = float(np.percentile(rel, 100 - k))
            pred = rel >= thresh
            iou, p, r = _iou_pr(pred, gt_mask)
            rows.append({
                "frame": ef, "top_k_pct": k, "iou": iou,
                "precision": p, "recall": r,
                "gt_pct": float(gt_mask.mean()) * 100,
                "thresh": thresh, "rel_mean": float(rel.mean()),
            })
            print(f"  frame {ef:3d} top-{k:>2d}% : "
                  f"IoU={iou:.3f}  P={p:.3f}  R={r:.3f}  GT={gt_mask.mean()*100:.1f}%")

    # 3. Aggregates
    print("\n[iou] per-top-k aggregates:")
    summary_lines = ["top_k_pct,iou_mean,iou_std,precision_mean,recall_mean,n_frames"]
    aggregates: list[tuple[int, float, float, float, float, int]] = []
    for k in top_ks:
        ks = [r for r in rows if r["top_k_pct"] == k]
        if not ks:
            continue
        ious = [r["iou"] for r in ks]
        ps = [r["precision"] for r in ks]
        rs = [r["recall"] for r in ks]
        agg = (k, float(np.mean(ious)), float(np.std(ious)),
               float(np.mean(ps)), float(np.mean(rs)), len(ks))
        aggregates.append(agg)
        summary_lines.append(",".join(str(x) for x in agg))
        print(f"  top-{k:>2d}% : IoU={agg[1]:.3f} +- {agg[2]:.3f} | "
              f"P={agg[3]:.3f}  R={agg[4]:.3f}  (n={agg[5]})")

    # 4. Persist outputs
    csv_per_frame = out_dir / f"iou_query_{clip_dir.name}_perframe.csv"
    with open(csv_per_frame, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    csv_summary = out_dir / f"iou_query_{clip_dir.name}.csv"
    csv_summary.write_text("\n".join(summary_lines) + "\n")

    txt_path = out_dir / f"iou_query_{clip_dir.name}.txt"
    with open(txt_path, "w") as f:
        f.write(f"H4 IoU report - clip={clip_dir.name}, query='{args.query}'\n")
        f.write(f"frames evaluated: {eval_frames}\n\n")
        f.write(f"{'top_k%':>7s}  {'IoU':>10s}  {'P':>6s}  {'R':>6s}  n\n")
        for (k, m, s, p, r, n) in aggregates:
            f.write(f"{k:>7d}  {m:.3f}+-{s:.3f}  {p:>.3f}  {r:>.3f}  {n}\n")

    # 5. Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ks = [a[0] for a in aggregates]
        means = [a[1] for a in aggregates]
        stds = [a[2] for a in aggregates]
        ax.errorbar(ks, means, yerr=stds, fmt="-o", color="#1f77b4",
                    capsize=4, label="DynLang-SLAM (rendered)")
        # Random-baseline reference for top-k%, with GT fraction averaged across eval frames
        if per_frame_gts:
            gt_frac_mean = float(np.mean([gt.mean() for gt in per_frame_gts]))
            random_iou = [
                gt_frac_mean * (k / 100.0)
                / (gt_frac_mean + (k / 100.0) - gt_frac_mean * (k / 100.0))
                for k in ks
            ]
            ax.plot(ks, random_iou, "--", color="grey",
                    label="Random baseline (top-k% AND uniform GT)")
        ax.set_xlabel("Top-k% pixels by relevancy intensity")
        ax.set_ylabel("IoU vs YOLO person mask")
        ax.set_title(f"H4 / {clip_dir.name}  -  query='{args.query}'  "
                     f"({len(eval_frames)} frames)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        plot_path = out_dir / f"iou_query_{clip_dir.name}.png"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        print(f"[iou] -> {plot_path}")
    except Exception as e:
        print(f"[iou] plotting failed: {e}")

    print(f"[iou] tables -> {csv_summary}, {csv_per_frame}, {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
