"""Figure: full DynLang-SLAM system filmstrip on H1.

Composes a 4-row x 5-column figure from existing per-frame PNGs in
results/figures/filmstrip_h1/:
  Row 1: Input RGB
  Row 2: SLAM render (from this frame's estimated pose)
  Row 3: Dynamic mask overlay (red = dynamic, kept out of loss)
  Row 4: Language relevancy heatmap for query "person"

This is the single integrated view that the abstract's "we unify
mapping + dynamic + language in one online pipeline" claim is making.

Usage:
    python scripts/figure_full_system.py
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
SRC = os.path.join(FIG_DIR, "filmstrip_h1")

FRAMES = [1, 15, 30, 60, 99]
ROWS = [
    ("rgb",                 "Input RGB"),
    ("render",              "Render (est. pose)"),
    ("mask",                "Dynamic mask"),
    ("lang_person-human",   "Lang query:\n``person''"),
]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load(slug, fid):
    path = os.path.join(SRC, f"frame_{fid:03d}_{slug}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    n_rows, n_cols = len(ROWS), len(FRAMES)
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(2.10 * n_cols + 0.6, 1.55 * n_rows + 0.3),
        constrained_layout=True,
    )

    for ci, fid in enumerate(FRAMES):
        axs[0, ci].set_title(f"frame {fid}", fontsize=9)
        for ri, (slug, _) in enumerate(ROWS):
            try:
                axs[ri, ci].imshow(_load(slug, fid))
            except FileNotFoundError as e:
                axs[ri, ci].text(0.5, 0.5, "missing",
                                 ha="center", va="center",
                                 transform=axs[ri, ci].transAxes,
                                 color="#888", fontsize=8)
                print(f"  skip: {e}")
            axs[ri, ci].set_xticks([]); axs[ri, ci].set_yticks([])
            for s in axs[ri, ci].spines.values():
                s.set_linewidth(0.4); s.set_edgecolor("#444")

    # Row labels on left side
    for ri, (_, label) in enumerate(ROWS):
        axs[ri, 0].set_ylabel(label, fontsize=9, fontweight="bold")

    fig.suptitle(
        "DynLang-SLAM full pipeline on BONN person\\_tracking: "
        "input, render, dynamic mask, and open-vocabulary "
        "language heatmap, across timesteps",
        fontsize=10, y=1.03,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_full_system.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
