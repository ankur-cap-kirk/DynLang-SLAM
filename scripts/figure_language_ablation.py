"""Figure 10: language-feature distillation ablation (broken vs fixed AE).

Reads per-frame language heatmaps from two H1 runs:

    results/figures/filmstrip_h1/             <- fixed AE (lang_dim=128, warmup=20)
    results/figures/filmstrip_h1_broken_ae/   <- broken AE (lang_dim=16, warmup=100)

The broken config was the original code -- on a 50-frame BONN run the AE
never reached its convergence threshold *and* the lang_loss never activated
(the gating waits for AE freeze, which only happened past the 3xwarmup
hard cap, after the run ended). The result: per-Gaussian lang_feats stay
at the zero-init, and queries fire on noise.

The fixed config (warmup=20, latent_dim=128, lower convergence threshold,
faster ramp) lets the AE freeze around frame 30-40 and the lang_loss
actually drives the lang_feats for the second half of the run.

Layout: 2 rows x 3 cols. Each col = a different query.
        Top row    = broken AE (firing on noise)
        Bottom row = fixed AE  (localizing on the right object)

Frame used: the last capture frame (99 by default), where both runs have
the most map coverage.

Usage:
    python scripts/figure_language_ablation.py

Prereq runs:
    python scripts/filmstrip.py                       # fixed AE (current code)
    DYNLANG_BROKEN_AE=1 DYNLANG_OUT_SUFFIX=broken_ae \\
        python scripts/filmstrip.py                   # broken AE
"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

FRAME = int(os.environ.get("DYNLANG_ABLATION_FRAME", "99"))

# Three queries chosen to highlight the difference: one CLIP-strong word
# (chair), one mid-difficulty (desk|table), one previously-failing
# (person|human).
QUERIES = [
    ("chair",                  "chair"),
    ("desk-table",             "desk / table"),
    ("person-human",           "person"),
]

DIR_FIXED  = os.path.join(FIG_DIR, "filmstrip_h1")
DIR_BROKEN = os.path.join(FIG_DIR, "filmstrip_h1_broken_ae")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load(d, frame, slug):
    path = os.path.join(d, f"frame_{frame:03d}_lang_{slug}.png")
    if not os.path.exists(path):
        sys.exit(f"[fig10] missing: {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    cols = len(QUERIES)
    fig, axs = plt.subplots(
        2, cols,
        figsize=(1.85 * cols + 0.6, 2 * 1.5 + 0.6),
        constrained_layout=True,
    )
    if cols == 1:
        axs = axs.reshape(2, 1)

    for ci, (slug, display) in enumerate(QUERIES):
        # Top: broken
        img_b = _load(DIR_BROKEN, FRAME, slug)
        axs[0, ci].imshow(img_b)
        axs[0, ci].set_xticks([]); axs[0, ci].set_yticks([])
        for s in axs[0, ci].spines.values():
            s.set_linewidth(0.4); s.set_edgecolor("#B22222")
        axs[0, ci].set_title(f"“{display}”", fontsize=9)
        if ci == 0:
            axs[0, ci].set_ylabel(
                "Pre-fix\nlatent=16, warmup=100\n(AE never freezes)",
                fontsize=8.5, color="#B22222", fontweight="bold")

        # Bottom: fixed
        img_f = _load(DIR_FIXED, FRAME, slug)
        axs[1, ci].imshow(img_f)
        axs[1, ci].set_xticks([]); axs[1, ci].set_yticks([])
        for s in axs[1, ci].spines.values():
            s.set_linewidth(0.4); s.set_edgecolor("#1f7a1f")
        if ci == 0:
            axs[1, ci].set_ylabel(
                "Post-fix\nlatent=16, warmup=20\n(AE freezes ≈frame 21)",
                fontsize=8.5, color="#1f7a1f", fontweight="bold")

    fig.suptitle(
        f"Language-feature distillation ablation on BONN person\\_tracking "
        f"(frame {FRAME}, prompt-ensembled relevancy)",
        fontsize=10, y=1.04,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_language_ablation.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
