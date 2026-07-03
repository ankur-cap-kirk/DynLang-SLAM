"""Figure 13: failure-cases panel (mandatory CVPR honesty section).

Three labeled examples of where DynLang-SLAM breaks. Important to
include — reviewers expect a limitations / failure-modes panel and
flag papers that omit one.

Cases shown:
  A. Synonym / color mismatch in CLIP queries
     H2 balloon\\_tracking, query "balloon|red_balloon".
     Actual balloon is *yellow*; CLIP's text embedding for "red balloon"
     drives the relevancy onto red-clothed person and red-tinted desk
     surfaces, not the balloon itself. Hard CLIP failure mode that no
     amount of distillation tuning can fix at the supervision side.

  B. Small-object coarse localization
     H1 person\\_tracking, query "keyboard".
     Keyboards are small (a few hundred pixels) and CLIP's averaged
     SAM-mask features wash them out. Heatmap fires broadly on the
     desk surface and floor instead of a tight peak.

  C. Map holes from sparse view coverage
     H1 person\\_tracking, canonical-pose render at frame 99.
     The canonical viewpoint includes regions the camera never observed
     directly; those pixels render as transparent/black because no
     Gaussians cover that frustum. Inherent to the explicit-representation
     SLAM tradeoff.

Usage:
    python scripts/figure_failure_cases.py
"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


CASES = [
    dict(
        path=os.path.join(FIG_DIR, "filmstrip_h2",
                          "frame_060_lang_balloon-red_balloon.png"),
        title="(A) Synonym / color mismatch",
        caption=("Query \"balloon | red\\_balloon\" on H2.\n"
                 "Actual balloon is yellow; CLIP fires on red-clothed person\n"
                 "and warm-toned surfaces."),
    ),
    dict(
        path=os.path.join(FIG_DIR, "filmstrip_h1",
                          "frame_060_lang_keyboard-computer_keyboard.png"),
        title="(B) Small-object coarse localization",
        caption=("Query \"keyboard\" on H1.\n"
                 "Sub-pixel-scale objects are washed out by SAM-mask\n"
                 "averaged CLIP features; heatmap fires broadly."),
    ),
    dict(
        path=os.path.join(FIG_DIR, "filmstrip_h1",
                          "frame_001_canonical.png"),
        title="(C) Map holes from sparse view coverage",
        caption=("Canonical-pose render at an early frame.\n"
                 "Regions outside the camera's frustum-history have no\n"
                 "Gaussians and render as black."),
    ),
]


def main():
    cols = len(CASES)
    fig, axs = plt.subplots(
        1, cols,
        figsize=(2.55 * cols + 0.3, 2.3),
        constrained_layout=True,
    )
    if cols == 1:
        axs = [axs]

    for ax, case in zip(axs, CASES):
        path = case["path"]
        if not os.path.exists(path):
            sys.exit(f"[fig13] missing: {path}")
        ax.imshow(np.asarray(Image.open(path).convert("RGB")))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor("#444")
        ax.set_title(case["title"], fontsize=10, fontweight="bold")
        # Caption beneath the image, italicized, smaller
        ax.set_xlabel(case["caption"], fontsize=7.8, style="italic",
                      labelpad=6, linespacing=1.3)

    fig.suptitle("Failure modes",
                 fontsize=11, fontweight="bold", y=1.06)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_failure_cases.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
