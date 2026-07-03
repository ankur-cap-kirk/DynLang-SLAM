"""Per-frame dynamic-mask overlay figure (paper Sec. 3.2 Dynamic Masking).

Shows what YOLOv8x-Seg + temporal filter actually detects on H1 BONN
person\\_tracking, frame by frame. Each cell = input RGB with the dynamic
mask painted red. Read together with the ablation figure
(figure_dynamic_ablation) so the reader sees both the mask itself
and the downstream effect on the Gaussian map.

Usage:
    python scripts/figure_dynamic_mask.py
"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

FRAMES = [int(s) for s in os.environ.get("DYNLANG_FRAMES", "1,15,30,60,99").split(",")]
DIR = os.path.join(FIG_DIR, "filmstrip_h1")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load(frame, sig):
    path = os.path.join(DIR, f"frame_{frame:03d}_{sig}.png")
    if not os.path.exists(path):
        sys.exit(f"[fig_dynmask] missing: {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    cols = len(FRAMES)
    fig, axs = plt.subplots(
        1, cols,
        figsize=(2.10 * cols + 0.35, 1.85),
        constrained_layout=True,
    )
    if cols == 1:
        axs = [axs]
    for ax, f in zip(axs, FRAMES):
        ax.imshow(_load(f, "mask"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor("#444")
        ax.set_title(f"frame {f}", fontsize=9)

    fig.suptitle(
        "YOLOv8-Seg + temporal-filter dynamic mask on H1, "
        "dilated $5{\\times}5$. Red overlay marks pixels excluded "
        "from the photometric loss.",
        fontsize=9.5, y=1.10,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_dynamic_mask.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
