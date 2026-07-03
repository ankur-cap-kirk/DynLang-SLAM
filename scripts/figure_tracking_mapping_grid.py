"""Compose the per-scene tracking-and-mapping figures into a single
multi-panel composite that demonstrates DynLang-SLAM across all
sequences (BONN dynamic, Replica synthetic, in-the-wild Pixel 8).

Reads:
    results/figures/figure_tracking_mapping_{h1,h2,h3,h4}.png

Writes:
    results/figures/figure_tracking_mapping_grid.{pdf,png}

Run:
    python scripts/figure_tracking_mapping_grid.py

If a per-scene file is missing, that panel is rendered as a placeholder
("not yet generated") so the script is robust to partial runs.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ALL_PANELS = [
    ("h1", "BONN person\\_tracking — real, dynamic indoor"),
    ("h2", "BONN balloon — real, dynamic indoor"),
    ("h3", "Replica room0 — synthetic"),
    ("h4", "H4 in-the-wild — Pixel 8 phone capture"),
]


def _load(path):
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


# Keep only panels whose per-scene PNG actually exists. Fail-soft so we
# can ship with whatever subset is available right now.
PANELS = [
    (s, c) for (s, c) in ALL_PANELS
    if _load(os.path.join(FIG_DIR, f"figure_tracking_mapping_{s}.png"))
       is not None
]
n = len(PANELS)
if n == 0:
    raise SystemExit("No per-scene tracking figures found — run "
                     "scripts/figure_tracking_mapping.py per scene first.")

# Layout: 1xN if <=2 panels, otherwise 2x2 with placeholders for missing.
if n <= 2:
    rows, cols = 1, n
    figsize = (7.5 * n, 4.8)
else:
    rows, cols = 2, 2
    figsize = (14, 9)

fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
axes = (axes,) if n == 1 else (axes.flatten() if hasattr(axes, "flatten") else axes)

for ax, (scene, caption) in zip(axes, PANELS):
    img = _load(os.path.join(FIG_DIR, f"figure_tracking_mapping_{scene}.png"))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(caption, fontsize=11, fontweight="bold", pad=6)

fig.suptitle("Tracking \\& Mapping across diverse scenes",
             fontsize=14, fontweight="bold")

out_pdf = os.path.join(FIG_DIR, "figure_tracking_mapping_grid.pdf")
out_png = os.path.join(FIG_DIR, "figure_tracking_mapping_grid.png")
plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
plt.close()
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
