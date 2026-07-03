"""Figure 9: dynamic-masking ablation (ON vs OFF).

Reads per-frame canonical-pose renders from two H1 runs:

    results/figures/filmstrip_h1/         <- dynamic ON  (default config)
    results/figures/filmstrip_h1_dyn_off/ <- dynamic OFF (DYNLANG_DYNAMIC_OFF=1)

Composes a 2-row x N-col figure showing the canonical-pose render at each
captured frame for both configurations. The contamination story:

    With dynamic masking OFF, the moving person bakes into the Gaussian map.
    From the canonical (frame-60) viewpoint you see ghost streaks where the
    person's silhouette was rendered at every frame -- the map is corrupted
    by transient geometry.

    With dynamic masking ON, those pixels are excluded from the mapping
    loss, the per-Gaussian Bayesian belief tags any leakage, and
    cleanup_contaminated() prunes the offenders. The canonical render is
    clean.

Usage:
    python scripts/figure_dynamic_ablation.py

Prereq runs:
    python scripts/filmstrip.py                                    # ON
    DYNLANG_DYNAMIC_OFF=1 DYNLANG_LANG=0 \\
        DYNLANG_OUT_SUFFIX=dyn_off python scripts/filmstrip.py     # OFF
"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Frames to compare. We *intentionally* drop frames 1 and 15 from the
# default: the canonical-pose viewpoint is locked to the frame-60 pose,
# and at frames 1/15 the dynamic-ON snapshot has very few Gaussians in
# that frustum so the panel is mostly black -- visually noisy without
# adding scientific content. Frames 30/60/99 all have real map coverage
# in both runs and make the ghost-vs-clean comparison legible.
FRAMES = [int(s) for s in os.environ.get("DYNLANG_FRAMES", "30,60,99").split(",")]

# Source dirs
DIR_ON  = os.path.join(FIG_DIR, "filmstrip_h1")
DIR_OFF = os.path.join(FIG_DIR, "filmstrip_h1_dyn_off")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load(d, frame, sig):
    path = os.path.join(d, f"frame_{frame:03d}_{sig}.png")
    if not os.path.exists(path):
        sys.exit(f"[fig9] missing: {path}\n"
                 f"  Run the appropriate filmstrip.py invocation first.")
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    cols = len(FRAMES)
    # Slightly larger cell size now that we have fewer cols
    fig, axs = plt.subplots(
        2, cols,
        figsize=(2.20 * cols + 0.85, 2 * 1.55 + 0.5),
        constrained_layout=True,
    )
    if cols == 1:
        axs = axs.reshape(2, 1)

    for ci, f in enumerate(FRAMES):
        # Top row: dynamic OFF
        img_off = _load(DIR_OFF, f, "canonical")
        axs[0, ci].imshow(img_off)
        axs[0, ci].set_xticks([]); axs[0, ci].set_yticks([])
        for s in axs[0, ci].spines.values():
            s.set_linewidth(0.4); s.set_edgecolor("#B22222")
        if ci == 0:
            axs[0, ci].set_ylabel("Dynamic\nmasking OFF",
                                  fontsize=9, fontweight="bold",
                                  color="#B22222")
        axs[0, ci].set_title(f"frame {f}", fontsize=8)

        # Bottom row: dynamic ON
        img_on = _load(DIR_ON, f, "canonical")
        axs[1, ci].imshow(img_on)
        axs[1, ci].set_xticks([]); axs[1, ci].set_yticks([])
        for s in axs[1, ci].spines.values():
            s.set_linewidth(0.4); s.set_edgecolor("#1f7a1f")
        if ci == 0:
            axs[1, ci].set_ylabel("Dynamic\nmasking ON\n(ours)",
                                  fontsize=9, fontweight="bold",
                                  color="#1f7a1f")

    fig.suptitle(
        "Dynamic-object masking ablation on BONN person\\_tracking "
        "(canonical-pose render across timesteps)",
        fontsize=10, y=1.04,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_dynamic_ablation.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
