"""Combined cross-scene figure composer.

Reads per-frame PNGs already saved by `scripts/filmstrip.py` (under
results/figures/filmstrip_<scene>/) and assembles a single unified
cross-scene figure suitable for a CVPR teaser or hero figure. NO SLAM
re-run -- pure file I/O + matplotlib.

Layouts (selectable via --layout):

    progress    [DEFAULT]
        3 rows (one per scene) x 5 cols (capture frames).
        Each cell = canonical-pose render. Tells the unified
        "map build-up across 3 datasets" story in one strip.

    summary
        3 rows x 5 cols. One representative frame per column
        per scene: Input / Render / Canonical / Dynamic mask /
        Language query. Single representative frame per scene
        (default = frame 60).

    full
        9 rows x 5 cols. For each scene: Input row + Render row +
        Canonical row, stacked. The "everything across all scenes"
        layout. Tall -- best for a full-page or appendix figure.

Usage:
    python scripts/figures_combined.py
    python scripts/figures_combined.py --layout summary
    python scripts/figures_combined.py --layout full
    python scripts/figures_combined.py --query desk    # for `summary`
"""
import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Scene metadata -- order in this dict = top-to-bottom row order in figure.
SCENES = [
    ("h3", "Replica room0",          "filmstrip_h3", "table"),
    ("h1", "BONN person\\_tracking", "filmstrip_h1", "desk"),
    ("h2", "BONN balloon",           "filmstrip_h2", "table"),
]

# Default capture frames -- must match scripts/filmstrip.py default.
FRAMES = [1, 15, 30, 60, 99]


def _load(path: str) -> np.ndarray:
    """Load a PNG to (H, W, 3) uint8."""
    if not os.path.exists(path):
        sys.exit(f"[figures_combined] missing: {path}\n"
                 f"  Run scripts/filmstrip.py first for the relevant scene.")
    img = np.asarray(Image.open(path).convert("RGB"))
    return img


def _set_cvpr_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, basename):
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{basename}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")


def _plot_grid(rows_arrays, row_labels, col_titles, suptitle, basename):
    """rows_arrays: list[list[ndarray]] -- outer = rows, inner = cols.
    Common helper for every layout below."""
    rows = len(rows_arrays)
    cols = len(rows_arrays[0])
    fig, axs = plt.subplots(
        rows, cols,
        figsize=(1.55 * cols + 0.7, 1.55 * rows + 0.4),
        constrained_layout=True,
    )
    if rows == 1:
        axs = np.array([axs])
    if cols == 1:
        axs = axs.reshape(rows, 1)
    for r in range(rows):
        for c in range(cols):
            ax = axs[r, c]
            ax.imshow(rows_arrays[r][c])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if r == 0 and col_titles is not None:
                ax.set_title(col_titles[c], fontsize=8)
            if c == 0 and row_labels is not None:
                ax.set_ylabel(row_labels[r], fontsize=9, fontweight="bold")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.02)
    _save(fig, basename)
    plt.close(fig)


# ---- Layouts -------------------------------------------------------------

def layout_progress(args):
    """3 rows (scenes) x 5 cols (frames). Cells = canonical-pose render."""
    rows_arrays = []
    row_labels = []
    for tag, display, dirname, _ in SCENES:
        d = os.path.join(FIG_DIR, dirname)
        rows_arrays.append([
            _load(os.path.join(d, f"frame_{f:03d}_canonical.png"))
            for f in FRAMES
        ])
        row_labels.append(display)
    col_titles = [f"frame {f}" for f in FRAMES]
    _plot_grid(
        rows_arrays, row_labels, col_titles,
        suptitle="DynLang-SLAM map build-up across 3 datasets "
                 "(canonical-pose render, same view for all timesteps)",
        basename="all_scenes_progress",
    )


def layout_summary(args):
    """3 rows (scenes) x 5 cols (signals at one rep frame).
    Signals: Input / Render / Canonical / Dynamic mask / Language query."""
    f = args.rep_frame
    q = args.query
    cols_signals = ["rgb", "render", "canonical", "mask", f"lang_{q}"]
    col_titles = ["Input", "Render @ this pose",
                  f"Render @ frame-60 pose", "Dynamic mask",
                  f"Language: '{q}'"]
    rows_arrays = []
    row_labels = []
    for tag, display, dirname, default_q in SCENES:
        d = os.path.join(FIG_DIR, dirname)
        # Use scene-default query unless --query overrides AND the file exists
        per_scene_q = q
        cand = os.path.join(d, f"frame_{f:03d}_lang_{q}.png")
        if not os.path.exists(cand):
            per_scene_q = default_q
            cand = os.path.join(d, f"frame_{f:03d}_lang_{default_q}.png")
        sig_files = [
            os.path.join(d, f"frame_{f:03d}_rgb.png"),
            os.path.join(d, f"frame_{f:03d}_render.png"),
            os.path.join(d, f"frame_{f:03d}_canonical.png"),
            os.path.join(d, f"frame_{f:03d}_mask.png"),
            cand,
        ]
        rows_arrays.append([_load(p) for p in sig_files])
        # Update column 5's label per-scene if scenes used different defaults
        if per_scene_q != q:
            # We can't easily make column titles per-row; skip the override
            pass
        row_labels.append(display)
    _plot_grid(
        rows_arrays, row_labels, col_titles,
        suptitle=f"DynLang-SLAM across 3 datasets at frame {f}: "
                 "tracking, mapping, dynamic masking, and language querying",
        basename=f"all_scenes_summary_frame{f}",
    )


def layout_full(args):
    """9 rows x 5 cols. For each scene, 3 rows (Input / Render / Canonical)."""
    rows_arrays = []
    row_labels = []
    for tag, display, dirname, _ in SCENES:
        d = os.path.join(FIG_DIR, dirname)
        for sig, label in [("rgb", "Input"),
                           ("render", "Render @ this pose"),
                           ("canonical", f"Render @ frame-60 pose")]:
            rows_arrays.append([
                _load(os.path.join(d, f"frame_{f:03d}_{sig}.png"))
                for f in FRAMES
            ])
            row_labels.append(f"{display}\n{label}")
    col_titles = [f"frame {f}" for f in FRAMES]
    _plot_grid(
        rows_arrays, row_labels, col_titles,
        suptitle="DynLang-SLAM tracking + mapping across 3 datasets",
        basename="all_scenes_full",
    )


def layout_everything(args):
    """15 rows x 5 cols. THE everything figure: for each of 3 scenes, 5 rows
    (Input / Render / Canonical / Dynamic mask / Language) across all 5
    capture frames. Language column uses the scene-default query."""
    rows_arrays = []
    row_labels = []
    for tag, display, dirname, default_q in SCENES:
        d = os.path.join(FIG_DIR, dirname)
        signals = [
            ("rgb",       "Input"),
            ("render",    "Render @ this pose"),
            ("canonical", "Render @ frame-60 pose"),
            ("mask",      "Dynamic mask"),
            (f"lang_{default_q}", f"Language: '{default_q}'"),
        ]
        for sig, label in signals:
            rows_arrays.append([
                _load(os.path.join(d, f"frame_{f:03d}_{sig}.png"))
                for f in FRAMES
            ])
            row_labels.append(f"{display}\n{label}")
    col_titles = [f"frame {f}" for f in FRAMES]
    _plot_grid(
        rows_arrays, row_labels, col_titles,
        suptitle="DynLang-SLAM: tracking, mapping, dynamic masking, "
                 "and language querying across 3 datasets",
        basename="all_scenes_everything",
    )


# ---- Driver --------------------------------------------------------------

LAYOUTS = {
    "progress": layout_progress,
    "summary": layout_summary,
    "full": layout_full,
    "everything": layout_everything,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="progress",
                   choices=sorted(LAYOUTS.keys()))
    p.add_argument("--rep-frame", type=int, default=60,
                   help="Representative frame for `summary` layout.")
    p.add_argument("--query", default="table",
                   help="Language query column for `summary` layout.")
    args = p.parse_args()

    _set_cvpr_style()
    print(f"[figures_combined] layout={args.layout}")
    LAYOUTS[args.layout](args)
    print("[done]")


if __name__ == "__main__":
    main()
