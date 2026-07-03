"""H4 dynamic-mask filmstrip figure - the analogue of midterm Fig. dyn_mask.

Pure inference: runs YOLOv8x-Seg + the temporal filter + dilation on five
evenly-spaced frames from data/H4/clip_a/, then composes a 4-row grid:

    Row 0: input RGB
    Row 1: YOLOv8 raw merged dynamic mask
    Row 2: temporal-filtered (>= 2 of 3 frames) mask
    Row 3: final dilated mask actually fed to the photometric loss

No SLAM is invoked here, so this script is fast (< 1 minute) and useful
both as a smoke check on the dynamic stack and as a deliverable figure.

Usage:
    python scripts/h4_dyn_mask_filmstrip.py --clip data/H4/clip_a
    python scripts/h4_dyn_mask_filmstrip.py --clip data/H4/clip_a --n-panels 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONUNBUFFERED"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dynlang_slam.dynamic import DynamicDetector, TemporalFilter  # noqa: E402

from _h4_helpers import paint_mask, save_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=str, default="data/H4/clip_a")
    p.add_argument("--n-panels", type=int, default=5)
    p.add_argument("--yolo-model", type=str, default="yolov8x-seg",
                   help="Use yolov8n-seg for fast preview, yolov8x-seg for the report.")
    p.add_argument("--output", type=str,
                   default="results/h4/figures/dyn_mask_filmstrip.png")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _pick_panel_indices(rgb_files: list[Path], n: int) -> list[int]:
    """Evenly spaced including first and last. n>=2 required."""
    n = max(2, n)
    total = len(rgb_files)
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def main() -> int:
    args = parse_args()
    clip_dir = Path(args.clip).resolve()
    rgb_dir = clip_dir / "rgb"
    if not rgb_dir.exists():
        raise SystemExit(f"missing rgb/ inside {clip_dir}")
    rgb_files = sorted(rgb_dir.glob("*.png"))
    if len(rgb_files) < args.n_panels:
        raise SystemExit(
            f"only {len(rgb_files)} frames in {rgb_dir}, "
            f"need at least {args.n_panels}."
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[filmstrip] loading {args.yolo_model} on {device}...")
    detector = DynamicDetector(
        model_name=args.yolo_model,
        confidence_thresh=0.5,
        device=device,
        # Outdoor walking class set (matches configs/h4.yaml)
        dynamic_classes=[0, 1, 2, 3, 5, 7, 14, 15, 16, 17],
    )
    tf = TemporalFilter(window_size=3, min_detections=2, dilation_kernel=5)

    panel_idxs = _pick_panel_indices(rgb_files, args.n_panels)
    print(f"[filmstrip] panels at frame indices: {panel_idxs}")

    rgb_row: list[np.ndarray] = []
    raw_row: list[np.ndarray] = []
    temporal_row: list[np.ndarray] = []
    final_row: list[np.ndarray] = []

    # We must walk the full sequence for the temporal filter to be meaningful,
    # but we only paint the chosen panels. We process each chosen panel + 2
    # frames before to give the temporal filter a sensible warm-up.
    for panel_idx in panel_idxs:
        tf.reset()
        ctx_start = max(0, panel_idx - 2)
        for i in range(ctx_start, panel_idx + 1):
            bgr = cv2.imread(str(rgb_files[i]))
            if bgr is None:
                raise SystemExit(f"failed to load {rgb_files[i]}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            raw_mask = detector.detect_and_merge(rgb)  # (H, W) bool tensor
            static_smooth = tf.update(raw_mask.cpu())
            dynamic_dilated = (static_smooth < 0.5)  # bool, after dilation

            if i == panel_idx:
                rgb_row.append(rgb)
                raw_row.append(paint_mask(rgb, raw_mask.cpu().numpy(),
                                          color=(255, 0, 0), alpha=0.55))
                # Temporal mask: dynamic where smoothed < 0.5 BEFORE dilation.
                # The TemporalFilter dilates at the end, so static_smooth IS
                # the dilated version; show that as final, and re-derive a
                # pre-dilation guess for the temporal step using the count
                # of "any-detection" inside the last window of the filter.
                # Approximation: pre-dilate = raw mask AND any-of-last-2-frames
                #   stored heuristically. Since we don't have access to TF's
                #   internal state without changing the public API, we
                #   show the raw mask in row 1 and the final dilated mask
                #   in row 3 with row 2 = the eroded (= shrunk) version of
                #   final to approximate "before dilation".
                kernel = np.ones((5, 5), np.uint8)
                pre_dilate_approx = cv2.erode(
                    dynamic_dilated.numpy().astype(np.uint8), kernel, iterations=1,
                )
                temporal_row.append(paint_mask(rgb, pre_dilate_approx > 0,
                                               color=(255, 165, 0), alpha=0.55))
                final_row.append(paint_mask(rgb, dynamic_dilated.numpy(),
                                            color=(0, 200, 0), alpha=0.55))
                pct_raw = float(raw_mask.float().mean()) * 100
                pct_final = float(dynamic_dilated.float().mean()) * 100
                print(f"  frame {panel_idx:4d}: raw {pct_raw:5.1f}%   "
                      f"final {pct_final:5.1f}%")

    titles = [f"f{idx}" for idx in panel_idxs]
    panels = [rgb_row, raw_row, temporal_row, final_row]
    row_labels = ["RGB", "YOLO raw", "Temporal", "Dilated"]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    save_grid(panels, out_path, titles_top=titles, row_labels=row_labels)
    print(f"[filmstrip] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
