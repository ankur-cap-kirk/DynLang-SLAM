"""Figure 11: open-vocabulary multi-query gallery.

A single row of language-relevancy heatmaps from the H1 BONN
person\\_tracking scene at frame 99 (latest map). Each column = one query.
Demonstrates the system's open-vocabulary breadth without re-running SLAM.

Usage:
    python scripts/figure_multi_query.py
"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Frame to display (latest available).
FRAME = int(os.environ.get("DYNLANG_GALLERY_FRAME", "99"))

# (slug, display label) pairs. Slug = file name suffix produced by
# filmstrip.py. Display = how it'll be titled in the figure.
QUERIES = [
    ("chair",                       "“chair”"),
    ("desk-table",                  "“desk / table”"),
    ("keyboard-computer_keyboard",  "“keyboard”"),
    ("monitor-screen-display",      "“monitor / screen”"),
    ("person-human",                "“person”"),
]

DIR = os.path.join(FIG_DIR, "filmstrip_h1")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load(slug):
    path = os.path.join(DIR, f"frame_{FRAME:03d}_lang_{slug}.png")
    if not os.path.exists(path):
        sys.exit(f"[fig11] missing: {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    cols = len(QUERIES)
    fig, axs = plt.subplots(
        1, cols,
        figsize=(2.10 * cols + 0.3, 1.95),
        constrained_layout=True,
    )
    if cols == 1:
        axs = [axs]

    for ax, (slug, display) in zip(axs, QUERIES):
        ax.imshow(_load(slug))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor("#444")
        ax.set_title(display, fontsize=10)

    fig.suptitle(
        f"Open-vocabulary queries on the BONN person\\_tracking map "
        f"(frame {FRAME}, prompt-ensembled relevancy)",
        fontsize=10, y=1.10,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_multi_query.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
