"""DynLang-SLAM pipeline diagram (v5 — cleaner layout, faithful structure).

Layout matches the author's original 4-column scheme:
    INPUT │ PER-FRAME PROCESSING │ MAP STATE │ OUTPUTS

Aesthetic refinements over earlier drafts:
    - Orthogonal arrow routing for long-range connections (feedback loop
      across the top, query stream across the bottom).
    - Larger embedded images that fit cleanly inside their boxes.
    - All labels horizontal; no rotated text.
    - Boxes resized to give breathing room around their content.
    - Single-arrowhead poly-lines for the multi-segment paths (no
      arrowheads in the middle of bent routes).

Run:
    python scripts/figure_pipeline_v2.py
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
FILM_DIR = os.path.join(FIG_DIR, "filmstrip_h1")
os.makedirs(FIG_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.4,
})

# ── 5-category palette matching the author's original ────────────────────
CLR_INPUT = "#DCE3F0"
CLR_GEOM  = "#FCD9B0"
CLR_DYN   = "#F4C5C8"
CLR_LANG  = "#D7E5BC"
CLR_MAP   = "#BDCEE7"
CLR_OUT   = "#DCD0E5"
EDGE      = "#262626"
EDGE_SOFT = "#888888"
ARROW     = "#333333"
ARROW_HOT = "#A52A2A"
ARROW_QRY = "#1F5C99"


# ── Image loader (slightly larger thumbnails) ─────────────────────────────
def _load(path, size=(360, 270)):
    if not os.path.exists(path):
        return Image.new("RGB", size, "#cccccc")
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    return img

IMG_RGB    = _load(os.path.join(FILM_DIR, "frame_060_rgb.png"))
IMG_DEPTH  = _load(os.path.join(FILM_DIR, "frame_060_depth_input.png"))
IMG_MASK   = _load(os.path.join(FILM_DIR, "frame_060_mask.png"))
IMG_RENDER = _load(os.path.join(FILM_DIR, "frame_060_render.png"))
IMG_CANON  = _load(os.path.join(FILM_DIR, "frame_060_canonical.png"))
IMG_BELIEF = _load(os.path.join(FILM_DIR, "frame_060_belief.png"))
IMG_RELEV  = _load(os.path.join(FILM_DIR, "frame_060_lang_person-human.png"))


# ── Drawing helpers ───────────────────────────────────────────────────────
def add_box(ax, xy, wh, color, *, lw=1.0, radius=0.04, zorder=1,
            edge=EDGE):
    box = FancyBboxPatch(
        xy, wh[0], wh[1],
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=color, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def title_in_box(ax, xy, wh, text, *, size=10, italic_sub=None):
    """Place a bold title near the top edge of a module box. Optional
    italic subtitle one line below."""
    cx = xy[0] + wh[0] / 2
    cy = xy[1] + wh[1] - 0.20
    ax.text(cx, cy, text,
            ha="center", va="center",
            fontsize=size, fontweight="bold", color=EDGE,
            family="DejaVu Sans", zorder=4)
    if italic_sub:
        ax.text(cx, cy - 0.30, italic_sub,
                ha="center", va="center",
                fontsize=size - 1.5, color="#444",
                fontstyle="italic", family="DejaVu Sans", zorder=4)


def add_image(ax, img, center, zoom=0.30, edge=EDGE_SOFT):
    cx, cy = center
    oi = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        oi, (cx, cy), frameon=True,
        bboxprops=dict(edgecolor=edge, lw=0.6),
        pad=0.04,
    )
    ax.add_artist(ab)


def add_text(ax, xy, text, *, fontsize=8.5, ha="center", va="center",
             weight="normal", color="#1a1a1a", italic=False, family=None):
    kw = {"ha": ha, "va": va, "fontsize": fontsize,
          "fontweight": weight, "color": color, "zorder": 5}
    if italic:
        kw["fontstyle"] = "italic"
    if family:
        kw["family"] = family
    ax.text(xy[0], xy[1], text, **kw)


def arrow(ax, p0, p1, *, color=ARROW, lw=1.1, style="-",
          head=8, zorder=3, rad=0.0):
    a = FancyArrowPatch(
        p0, p1,
        arrowstyle=f"-|>,head_length={head},head_width={head*0.55}",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw, color=color, linestyle=style,
        shrinkA=2, shrinkB=2, zorder=zorder,
    )
    ax.add_patch(a)


def poly_arrow(ax, pts, *, color=ARROW, lw=1.1, style="-", head=8,
               zorder=3):
    """Multi-segment arrow: draws straight line segments through `pts`
    (a list of (x,y) points), then a final arrow head onto the last
    point. Avoids the FancyArrowPatch artefact of putting arrowheads
    on every segment of a bent route."""
    if len(pts) < 2:
        return
    xs = [p[0] for p in pts[:-1]]
    ys = [p[1] for p in pts[:-1]]
    # Add the second-to-last point twice to create a stub for the arrow
    xs.append(pts[-1][0])
    ys.append(pts[-1][1])
    ax.plot(xs, ys, color=color, lw=lw, linestyle=style, zorder=zorder,
            solid_capstyle="round")
    # Arrow head: from a tiny offset before the final point onto the final
    p_pre, p_last = pts[-2], pts[-1]
    arrow(ax, p_pre, p_last, color=color, lw=lw, style=style, head=head,
          zorder=zorder)


def label_arrow(ax, midpt, text, *, fontsize=7.5, color="#333",
                italic=True):
    add_text(ax, midpt, text, fontsize=fontsize, color=color,
             italic=italic)


# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(17, 9.6))
ax.set_xlim(0, 17)
ax.set_ylim(0, 9.6)
ax.set_aspect("equal")
ax.axis("off")

# ── Column headers (letter-spaced manually) ───────────────────────────────
HDR_Y = 9.10
for cx, txt in [
    (1.55, "INPUT"),
    (5.75, "PER-FRAME PROCESSING"),
    (10.30, "MAP STATE"),
    (14.45, "OUTPUTS"),
]:
    ax.text(cx, HDR_Y, " ".join(list(txt)),
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            family="DejaVu Sans", color="#222", zorder=10)

# horizontal rule below headers (kept tight to header text so it doesn't
# clash with the feedback-loop arrow that runs near the top of the figure)
ax.plot([0.30, 16.70], [HDR_Y - 0.10, HDR_Y - 0.10],
        color="#bbb", lw=0.6, zorder=0)


# ════════════════════════════════════════════════════════════════════════
# INPUT column  (x = 0.40 .. 2.70)
# ════════════════════════════════════════════════════════════════════════
INP_RGBD = (0.40, 6.20, 2.30, 2.40)
add_box(ax, INP_RGBD[:2], INP_RGBD[2:], CLR_INPUT)
title_in_box(ax, INP_RGBD[:2], INP_RGBD[2:], "RGB-D frame",
             italic_sub=r"$I_t,\ D_t$")
add_image(ax, IMG_RGB,   (1.05, 7.25), zoom=0.18)
add_image(ax, IMG_DEPTH, (2.05, 7.25), zoom=0.18)
add_text(ax, (1.05, 6.55), "RGB",   fontsize=7.8, color="#333")
add_text(ax, (2.05, 6.55), "Depth", fontsize=7.8, color="#333")

INP_POSE = (0.40, 4.55, 2.30, 1.20)
add_box(ax, INP_POSE[:2], INP_POSE[2:], CLR_INPUT)
title_in_box(ax, INP_POSE[:2], INP_POSE[2:], "Prev. pose",
             italic_sub=r"$T^{c\to w}_{t-1}$")
add_text(ax, (1.55, 4.85), r"$\in SE(3)$", fontsize=8.5, color="#333")

INP_QRY = (0.40, 1.85, 2.30, 1.55)
add_box(ax, INP_QRY[:2], INP_QRY[2:], CLR_INPUT)
title_in_box(ax, INP_QRY[:2], INP_QRY[2:], "Text query")
add_text(ax, (1.55, 2.40), r'\textit{``a chair''}',
         fontsize=10, italic=True, color="#222")
add_text(ax, (1.55, 2.05), "user-supplied", fontsize=7.5, color="#555")


# ════════════════════════════════════════════════════════════════════════
# PER-FRAME PROCESSING column  (x = 3.40 .. 6.10)
# ════════════════════════════════════════════════════════════════════════
PFP_TRK = (3.40, 6.95, 2.70, 1.65)
add_box(ax, PFP_TRK[:2], PFP_TRK[2:], CLR_GEOM)
title_in_box(ax, PFP_TRK[:2], PFP_TRK[2:], "Tracker",
             italic_sub="photometric L-M (50 it)")
add_text(ax, (4.75, 7.40), "alpha-gated", fontsize=8, color="#333")
add_text(ax, (4.75, 7.15), r"output: $T_t^{c\to w}$",
         fontsize=8, color="#333")

PFP_DYN = (3.40, 5.05, 2.70, 1.65)
add_box(ax, PFP_DYN[:2], PFP_DYN[2:], CLR_DYN)
title_in_box(ax, PFP_DYN[:2], PFP_DYN[2:], "Dynamic detector",
             italic_sub="YOLOv8x-Seg + temporal")
add_image(ax, IMG_MASK, (4.75, 5.55), zoom=0.20)

PFP_LNG = (3.40, 3.15, 2.70, 1.65)
add_box(ax, PFP_LNG[:2], PFP_LNG[2:], CLR_LANG)
title_in_box(ax, PFP_LNG[:2], PFP_LNG[2:], "Language extraction",
             italic_sub="CLIP-L/14 + SAM2")
add_text(ax, (4.75, 3.55),
         r"$\Phi_t \in \mathbb{R}^{H\times W\times 768}$",
         fontsize=8.5)
add_text(ax, (4.75, 3.30),
         "every K keyframes", fontsize=7.5, color="#555")

PFP_AE = (3.40, 1.30, 2.70, 1.65)
add_box(ax, PFP_AE[:2], PFP_AE[2:], CLR_LANG)
title_in_box(ax, PFP_AE[:2], PFP_AE[2:], "Autoencoder",
             italic_sub=r"$E_\theta:\,768\!\to\!16$")
add_text(ax, (4.75, 1.85), r"$D_\psi:\,16\!\to\!768$",
         fontsize=8.5, italic=True, color="#444")
add_text(ax, (4.75, 1.55),
         "scene-adaptive · frozen post warm-up",
         fontsize=7.3, color="#555")


# ════════════════════════════════════════════════════════════════════════
# MAP STATE column  (x = 7.40 .. 13.20)
# ════════════════════════════════════════════════════════════════════════
MS_MAP = (7.40, 6.95, 2.90, 1.65)
add_box(ax, MS_MAP[:2], MS_MAP[2:], CLR_GEOM)
title_in_box(ax, MS_MAP[:2], MS_MAP[2:], "Mapper",
             italic_sub="GS rasterize + L-M (60 it)")
add_text(ax, (8.85, 7.20),
         "60 iters/KF · sliding window 5",
         fontsize=8, color="#333")

MS_GM = (7.40, 1.30, 5.80, 5.30)
add_box(ax, MS_GM[:2], MS_GM[2:], CLR_MAP)
title_in_box(ax, MS_GM[:2], MS_GM[2:], "Gaussian Map",
             italic_sub="map state")

# Two embedded images side-by-side, matched in size
add_image(ax, IMG_CANON,  (9.10, 4.65), zoom=0.36)
add_image(ax, IMG_BELIEF, (11.65, 4.65), zoom=0.36)
add_text(ax, (9.10, 3.30), "canonical render",
         fontsize=7.5, color="#444", italic=True)
add_text(ax, (11.65, 3.30), "belief overlay",
         fontsize=7.5, color="#444", italic=True)

# State equation
add_text(ax, (10.30, 2.40),
         r"$\mathcal{G}=\{\mu_g,R_g,S_g,\alpha_g,c_g,f_{\mathrm{lang},g}\}"
         r"\ +\ b_g \in [0,1]$",
         fontsize=8.7)
add_text(ax, (10.30, 1.95),
         r"per-Gaussian dynamic belief · cleanup every 20 fr · prune $b_g\!>\!0.6$",
         fontsize=7.3, color="#555", italic=True)


# ════════════════════════════════════════════════════════════════════════
# OUTPUTS column  (x = 13.85 .. 16.70)
# ════════════════════════════════════════════════════════════════════════
OUT_TRAJ = (13.85, 6.95, 2.85, 1.65)
add_box(ax, OUT_TRAJ[:2], OUT_TRAJ[2:], CLR_OUT)
title_in_box(ax, OUT_TRAJ[:2], OUT_TRAJ[2:], "Camera trajectory")
add_text(ax, (15.275, 7.45), r"$\{T^{c\to w}_t\}_{t=1}^N$",
         fontsize=10.5, weight="bold")
add_text(ax, (15.275, 7.05),
         "Umeyama-aligned · multi-seed",
         fontsize=7.3, color="#555")

OUT_NV = (13.85, 5.05, 2.85, 1.70)
add_box(ax, OUT_NV[:2], OUT_NV[2:], CLR_OUT)
title_in_box(ax, OUT_NV[:2], OUT_NV[2:], "Novel-view render",
             italic_sub=r"$\hat I,\ \hat D$")
add_image(ax, IMG_RENDER, (15.275, 5.55), zoom=0.17)

OUT_DC = (13.85, 3.15, 2.85, 1.70)
add_box(ax, OUT_DC[:2], OUT_DC[2:], CLR_OUT)
title_in_box(ax, OUT_DC[:2], OUT_DC[2:], "Dynamic-clean map",
             italic_sub=r"prune $b_g>\theta$")
add_image(ax, IMG_CANON, (15.275, 3.65), zoom=0.17)

OUT_LR = (13.85, 1.30, 2.85, 1.65)
add_box(ax, OUT_LR[:2], OUT_LR[2:], CLR_OUT)
title_in_box(ax, OUT_LR[:2], OUT_LR[2:], "Language relevancy",
             italic_sub=r"$r_g(q)=\mathrm{softmax}(\tau\langle f,q\rangle)$")
add_image(ax, IMG_RELEV, (15.275, 1.78), zoom=0.17)


# ════════════════════════════════════════════════════════════════════════
# ARROWS — orthogonal routing for long paths, short diagonals locally
# ════════════════════════════════════════════════════════════════════════

# ── INPUT → PER-FRAME (all elbowed / orthogonal) ────────────────────────
# RGB-D → Tracker (init): straight horizontal at top
arrow(ax, (2.70, 7.78), (3.40, 7.78), lw=1.2)
label_arrow(ax, (3.05, 7.95), "init")

# RGB-D → Dynamic detector (RGB): elbow down
poly_arrow(ax, [(2.70, 6.40), (3.05, 6.40), (3.05, 5.85), (3.40, 5.85)],
           lw=1.2)

# RGB-D → Language extraction: elbow further down (offset bus column 3.20)
poly_arrow(ax, [(2.70, 6.25), (3.20, 6.25), (3.20, 3.95), (3.40, 3.95)],
           lw=1.2)

# Prev. pose → Tracker (prior): elbow up
poly_arrow(ax, [(2.70, 5.15), (3.05, 5.15), (3.05, 7.30), (3.40, 7.30)],
           lw=1.2)
label_arrow(ax, (2.85, 5.40), "prior", fontsize=7.0)

# Text query → Autoencoder (query stream, blue): elbow up slightly
poly_arrow(ax, [(2.70, 2.60), (3.05, 2.60), (3.05, 2.10), (3.40, 2.10)],
           lw=1.2, color=ARROW_QRY)
label_arrow(ax, (2.85, 2.78), "CLIP-text",
            fontsize=7, color=ARROW_QRY)

# ── PER-FRAME internal: Language extraction → Autoencoder ──────────────
arrow(ax, (4.75, 3.15), (4.75, 2.95), lw=1.0)
label_arrow(ax, (5.20, 3.05),
            r"$\Phi_t\!\in\!\mathbb{R}^{768}$",
            fontsize=7)

# ── PER-FRAME → MAP STATE ────────────────────────────────────────────────
# Tracker → Mapper (T_t)
arrow(ax, (6.10, 7.78), (7.40, 7.78), lw=1.4)
label_arrow(ax, (6.75, 7.95), r"$T_t^{c\to w}$",
            fontsize=8, italic=False)

# Dynamic detector → Mapper (mask M_t, red)
poly_arrow(ax, [(6.10, 6.30), (6.75, 6.30), (6.75, 7.20), (7.40, 7.20)],
           color=ARROW_HOT, lw=1.4)
label_arrow(ax, (6.50, 6.55), r"mask $M_t$",
            fontsize=7, color=ARROW_HOT)

# Dynamic detector → Gaussian Map (belief b+=, red dashed)
arrow(ax, (6.10, 5.55), (7.40, 5.55), lw=1.2, style="--",
      color=ARROW_HOT)
label_arrow(ax, (6.75, 5.70), r"belief $b\!+\!=$",
            fontsize=7, color=ARROW_HOT)

# Autoencoder → Gaussian Map (f_lang)
arrow(ax, (6.10, 2.10), (7.40, 2.10), lw=1.2)
label_arrow(ax, (6.75, 2.30),
            r"$f_{\mathrm{lang},g}\!\in\!\mathbb{R}^{16}$",
            fontsize=7)

# Mapper ↔ Gaussian Map (grad / state, single double-headed vertical arrow)
ax.annotate(
    "", xy=(8.85, 6.65), xytext=(8.85, 6.95),
    arrowprops=dict(arrowstyle="<->", lw=1.2, color=ARROW),
    zorder=3,
)
label_arrow(ax, (9.55, 6.80), "grad / state", fontsize=7)

# ── Top feedback loop: Gaussian Map → Tracker (rendered Î, D̂) ─────────
# Path runs at y=8.80, label sits BELOW the path (between the path and
# Tracker top at y=8.60) so it is clear of both the column header text
# and the rule line.
poly_arrow(ax,
           [(11.20, 6.60), (11.20, 8.80), (4.75, 8.80), (4.75, 8.60)],
           lw=1.0, style="--", color=ARROW)
add_text(ax, (8.00, 8.70),
         r"rendered $\hat I,\hat D$ (feedback)",
         fontsize=7.5, italic=True, color="#444")

# ── MAP STATE → OUTPUTS ──────────────────────────────────────────────────
# Camera trajectory (poses)
arrow(ax, (13.20, 7.78), (13.85, 7.78), lw=1.2)
label_arrow(ax, (13.50, 7.95), "poses")

# Novel-view render (rasterize)
arrow(ax, (13.20, 5.45), (13.85, 5.45), lw=1.2)
label_arrow(ax, (13.50, 5.62), "rasterize", fontsize=7)

# Dynamic-clean map (b > θ)
arrow(ax, (13.20, 3.55), (13.85, 3.55), lw=1.2)
label_arrow(ax, (13.50, 3.72), r"$b\!>\!\theta$",
            fontsize=7, color=ARROW_HOT)

# ── Bottom query stream: Autoencoder → Language relevancy (blue) ────────
poly_arrow(ax,
           [(4.75, 1.30), (4.75, 0.85), (15.275, 0.85), (15.275, 1.30)],
           lw=1.1, color=ARROW_QRY)
add_text(ax, (10.00, 0.62),
         r"$q\in\mathbb{R}^{16}$  (text-query embedding)",
         fontsize=7.5, italic=True, color=ARROW_QRY)


# ════════════════════════════════════════════════════════════════════════
# LEGEND — bottom strip, centered
# ════════════════════════════════════════════════════════════════════════
LEG_Y = 0.18
swatches = [
    (CLR_INPUT, "Input / query"),
    (CLR_GEOM,  "Geometric module"),
    (CLR_DYN,   "Dynamic-object module"),
    (CLR_LANG,  "Language module"),
    (CLR_MAP,   "Map state"),
    (CLR_OUT,   "Output"),
]
sw_w, sw_h = 0.32, 0.28
spacing = 2.55
total_w = (len(swatches) - 1) * spacing + sw_w
start_x = (17 - total_w) / 2

for i, (col, txt) in enumerate(swatches):
    x = start_x + i * spacing
    box = FancyBboxPatch((x, LEG_Y), sw_w, sw_h,
                         boxstyle="round,pad=0,rounding_size=0.03",
                         linewidth=0.6, edgecolor=EDGE,
                         facecolor=col, zorder=5)
    ax.add_patch(box)
    ax.text(x + sw_w + 0.10, LEG_Y + sw_h / 2, txt,
            fontsize=8, ha="left", va="center", color="#222",
            family="DejaVu Sans", zorder=5)


# ── Save ──────────────────────────────────────────────────────────────────
out_pdf = os.path.join(FIG_DIR, "figure_pipeline_v2.pdf")
out_png = os.path.join(FIG_DIR, "figure_pipeline_v2.png")
plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=240)
plt.close()
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
