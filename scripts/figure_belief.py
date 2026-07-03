"""Figure 12: per-Gaussian Bayesian dynamic-belief evolution over time.

Reads the per-frame belief overlays produced by the patched filmstrip.py
(saved as `frame_NNN_belief.png` once `_render_belief_field` is wired in)
and composes a 2-row x N-col figure:

    Row 1: input RGB at each capture frame
    Row 2: belief field rendered at the same frame, colormap-overlaid on RGB

Tells the story: at frame 1 every Gaussian has belief 0 (no prior contam.
evidence), and as the person walks across the scene the Gaussians their
silhouette projects onto accumulate belief mass, eventually exceeding
the cleanup threshold and being pruned from the static map.

Usage:
    python scripts/figure_belief.py
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
        sys.exit(f"[fig12] missing: {path}\n"
                 f"  Re-run filmstrip.py after the belief-rendering patch "
                 f"to produce belief overlays.")
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    cols = len(FRAMES)
    fig, axs = plt.subplots(
        2, cols,
        figsize=(1.85 * cols + 0.7, 2 * 1.45 + 0.4),
        constrained_layout=True,
    )
    if cols == 1:
        axs = axs.reshape(2, 1)

    for ci, f in enumerate(FRAMES):
        # Top: input RGB
        axs[0, ci].imshow(_load(f, "rgb"))
        axs[0, ci].set_xticks([]); axs[0, ci].set_yticks([])
        for s in axs[0, ci].spines.values():
            s.set_linewidth(0.4); s.set_edgecolor("#444")
        axs[0, ci].set_title(f"frame {f}", fontsize=9)
        if ci == 0:
            axs[0, ci].set_ylabel("Input", fontsize=9, fontweight="bold")

        # Bottom: belief overlay
        axs[1, ci].imshow(_load(f, "belief"))
        axs[1, ci].set_xticks([]); axs[1, ci].set_yticks([])
        for s in axs[1, ci].spines.values():
            s.set_linewidth(0.4); s.set_edgecolor("#B22222")
        if ci == 0:
            axs[1, ci].set_ylabel("Per-Gaussian\ndynamic\\_belief $b$",
                                  fontsize=9, fontweight="bold",
                                  color="#B22222")

    fig.suptitle(
        "Bayesian dynamic-belief field $b\\in[0,1]$ on H1 BONN person\\_tracking. "
        "Gaussians the moving person projects onto accumulate belief; "
        "$b > \\theta$ are pruned at cleanup intervals.",
        fontsize=9.5, y=1.06,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_belief.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
