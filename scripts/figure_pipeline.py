"""DynLang-SLAM system / pipeline diagram (Figure 2 of the paper).

Pure-matplotlib block diagram. Cleaner v2 layout: orthogonal arrow routing,
no curved-line label collisions, clear forward / feedback / query streams.

Layout (4 horizontal bands, each band is one row of boxes):

    INPUT       PER-FRAME        MAP           OUTPUTS
                PROCESSING       STATE
    ---------   --------------   -----------   --------------
    RGB-D ───►  Tracker ──────►  ┐                Trajectory
                                 │
    Prev pose───┘                ▼
                Dynamic ──mask──►Mapper ◄──────►  Novel-view
                detector            │             render
                ▲                   │
                │                   ▼          Dynamic-clean
                Language       Gaussian ──────►  map
                extract.───►   Map state
                       │           │
                Auto-  │           │             Lang
                encoder────────────┴─query─►     relevancy
                 ▲
                 │
    Text query ──┘

Run:
    python scripts/figure_pipeline.py
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -- Palette: muted, paper-friendly. Distinct hue per module class. -------
CLR_INPUT    = "#E8EEF7"
CLR_TRACK    = "#FFE7C2"
CLR_DYNAMIC  = "#FFD0CC"
CLR_LANG     = "#D9E8D2"
CLR_MAP      = "#CFE0F0"
CLR_OUTPUT   = "#EFE5F5"
EDGE         = "#333333"
ARROW_C      = "#444444"
ARROW_HOT    = "#B22222"
ARROW_QUERY  = "#1f77b4"

LW_BOX = 0.9
FS_TITLE  = 10
FS_BODY   = 8.5
FS_TENSOR = 7.5


def add_box(ax, x, y, w, h, label, color, *, fontweight="bold"):
    """Rounded rectangle with a 2-line label (title/n/body)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.020,rounding_size=0.05",
        linewidth=LW_BOX, edgecolor=EDGE, facecolor=color, zorder=2,
    )
    ax.add_patch(box)
    if "\n" in label:
        title, body = label.split("\n", 1)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center",
                fontsize=FS_TITLE, fontweight=fontweight, zorder=3)
        ax.text(x + w/2, y + h*0.25, body, ha="center", va="center",
                fontsize=FS_BODY, style="italic", color="#333", zorder=3)
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=FS_TITLE, fontweight=fontweight, zorder=3)


def arrow(ax, p0, p1, *, color=ARROW_C, lw=1.1, ls="-", style="->"):
    """Straight or polyline arrow between two points."""
    a = FancyArrowPatch(
        p0, p1,
        arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, linestyle=ls, zorder=1,
    )
    ax.add_patch(a)


def ortho_arrow(ax, p0, p1, *, via_x=None, via_y=None,
                color=ARROW_C, lw=1.1, ls="-"):
    """L-shaped (right-angle) connector. Specify either via_x (vertical
    leg first) or via_y (horizontal leg first). Tip arrow at p1."""
    x0, y0 = p0
    x1, y1 = p1
    if via_x is not None:
        # horizontal to via_x then vertical to y1
        ax.plot([x0, via_x, via_x], [y0, y0, y1], color=color, lw=lw, ls=ls,
                zorder=1, solid_capstyle="round")
        arrow(ax, (via_x, y1), (x1, y1), color=color, lw=lw, ls=ls)
    elif via_y is not None:
        ax.plot([x0, x0, x1], [y0, via_y, via_y], color=color, lw=lw, ls=ls,
                zorder=1, solid_capstyle="round")
        arrow(ax, (x1, via_y), (x1, y1), color=color, lw=lw, ls=ls)
    else:
        arrow(ax, p0, p1, color=color, lw=lw, ls=ls)


def label(ax, x, y, txt, *, color="#333", style="italic", fs=FS_TENSOR):
    """White-bg label so it sits on top of any background line/arrow."""
    ax.text(x, y, txt, ha="center", va="center",
            fontsize=fs, color=color, style=style,
            bbox=dict(facecolor="white", edgecolor="none",
                      pad=1.2, alpha=0.92),
            zorder=4)


def _swatch(c, t):
    return Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                  markeredgecolor=EDGE, markersize=11, label=t)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 6.0), constrained_layout=False)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)
    ax.set_axis_off()

    # ===== Vertical band guides ==========================================
    for x in [2.55, 5.85, 9.30]:
        ax.plot([x, x], [0.45, 5.50], color="#cccccc", lw=0.5, ls=":",
                zorder=0)

    # ===== Column headers ================================================
    headers = [
        (1.20, 5.95, "I N P U T"),
        (4.20, 5.95, "P E R - F R A M E   P R O C E S S I N G"),
        (7.55, 5.95, "M A P   S T A T E"),
        (11.10, 5.95, "O U T P U T S"),
    ]
    for x, y, t in headers:
        ax.text(x, y, t, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#666",
                family="serif")

    # =====================================================================
    # Boxes
    # =====================================================================
    # Inputs (col 1)
    BX_RGBD  = (0.30, 4.65, 1.95, 0.85);  add_box(ax, *BX_RGBD,
        "RGB-D frame\n$I_t,\\;D_t$", CLR_INPUT)
    BX_POSE  = (0.30, 3.45, 1.95, 0.85);  add_box(ax, *BX_POSE,
        "Prev. pose\n$T_{t-1}^{c\\to w}$", CLR_INPUT)
    BX_QUERY = (0.30, 0.55, 1.95, 0.85);  add_box(ax, *BX_QUERY,
        "Text query\n\"a chair\"", CLR_INPUT)

    # Per-frame processing (col 2)
    BX_TRACK = (3.00, 4.65, 2.40, 0.85);  add_box(ax, *BX_TRACK,
        "Tracker\nphotometric L-M (50 it)", CLR_TRACK)
    BX_DYN   = (3.00, 3.45, 2.40, 0.85);  add_box(ax, *BX_DYN,
        "Dynamic detector\nYOLOv8x-seg + temporal", CLR_DYNAMIC)
    BX_LANG  = (3.00, 2.20, 2.40, 0.85);  add_box(ax, *BX_LANG,
        "Language extraction\nCLIP-L/14 $\\circ$ SAM2", CLR_LANG)
    BX_AE    = (3.00, 0.95, 2.40, 0.85);  add_box(ax, *BX_AE,
        "Autoencoder\n768$\\,\\to\\,$16", CLR_LANG)

    # Map state (col 3)
    BX_MAP   = (6.30, 4.65, 2.55, 0.85);  add_box(ax, *BX_MAP,
        "Mapper\nGS rasterize $+$ L-M (60 it)", CLR_TRACK)
    BX_GMAP  = (6.30, 1.95, 2.55, 2.05);  add_box(ax, *BX_GMAP,
        "Gaussian Map\n$\\{\\mu,\\,R,\\,S,\\,\\alpha,\\,c,\\,f_{\\!lang}\\}$"
        "\n$+$ dynamic\\_belief $b\\!\\in\\![0,1]$",
        CLR_MAP, fontweight="bold")

    # Outputs (col 4)
    BX_TRAJ  = (9.70, 4.85, 2.95, 0.75);  add_box(ax, *BX_TRAJ,
        "Camera trajectory\n$\\{T_t^{c\\to w}\\}_{t=1}^N$", CLR_OUTPUT)
    BX_REND  = (9.70, 3.65, 2.95, 0.75);  add_box(ax, *BX_REND,
        "Novel-view render\n$\\hat{I},\\;\\hat{D}$", CLR_OUTPUT)
    BX_CLEAN = (9.70, 2.45, 2.95, 0.75);  add_box(ax, *BX_CLEAN,
        "Dynamic-clean map\nGaussians w/ $b\\!>\\!\\theta$ pruned", CLR_OUTPUT)
    BX_REL   = (9.70, 0.55, 2.95, 0.85);  add_box(ax, *BX_REL,
        "Language relevancy\n$r(g) = \\sigma(\\tau\\,\\langle f, q\\rangle)$",
        CLR_OUTPUT)

    # =====================================================================
    # Arrows (forward flow). Compute endpoints from box geometry to keep
    # connections precise even if we resize boxes later.
    # =====================================================================
    def right_mid(b):  return (b[0] + b[2], b[1] + b[3]/2)
    def left_mid(b):   return (b[0],          b[1] + b[3]/2)
    def bot_mid(b):    return (b[0] + b[2]/2, b[1])
    def top_mid(b):    return (b[0] + b[2]/2, b[1] + b[3])

    # ----- INPUT -> PROCESSING -----
    arrow(ax, right_mid(BX_RGBD), left_mid(BX_TRACK))
    # Prev pose -> Tracker: orthogonal elbow (right then up) so it does
    # not cross the RGB-D -> Tracker arrow at a diagonal.
    pose_out = right_mid(BX_POSE)
    track_in_lower = (BX_TRACK[0], BX_TRACK[1] + BX_TRACK[3]*0.30)
    pose_elbow_x = 2.65
    ortho_arrow(ax, pose_out, track_in_lower, via_x=pose_elbow_x)
    label(ax, pose_elbow_x + 0.18,
              (pose_out[1] + track_in_lower[1]) / 2,
          "init")

    # RGB tap: drop a vertical dashed feed on the right side of the
    # RGB-D box so Dynamic + Language can pick off the same RGB stream.
    rgb_tap_x = BX_RGBD[0] + BX_RGBD[2] - 0.20
    rgb_tap_top = BX_RGBD[1]                 # bottom of RGB-D box
    rgb_tap_bot = BX_LANG[1] + BX_LANG[3]/2  # mid of language box
    ax.plot([rgb_tap_x, rgb_tap_x], [rgb_tap_top, rgb_tap_bot],
            color="#888", lw=0.9, ls="--", zorder=0)
    # Tap to Dynamic (mid-right)
    arrow(ax, (rgb_tap_x, BX_DYN[1] + BX_DYN[3]/2),
              left_mid(BX_DYN), color="#666", lw=0.9, ls="--")
    # Tap to Language (mid-right)
    arrow(ax, (rgb_tap_x, BX_LANG[1] + BX_LANG[3]/2),
              left_mid(BX_LANG), color="#666", lw=0.9, ls="--")
    label(ax, rgb_tap_x + 0.30, (BX_RGBD[1] + BX_DYN[1] + BX_DYN[3])/2,
          "RGB", color="#777")

    # ----- PROCESSING -> MAP -----
    # Tracker -> Mapper (pose)
    arrow(ax, right_mid(BX_TRACK), left_mid(BX_MAP), lw=1.3)
    label(ax, (right_mid(BX_TRACK)[0] + left_mid(BX_MAP)[0])/2,
              (right_mid(BX_TRACK)[1] + left_mid(BX_MAP)[1])/2 + 0.16,
          "$T_t^{c\\to w}$")

    # Dynamic -> Mapper (mask).  Route over the top of the Map state band:
    # leg right then up into the Mapper's bottom edge.
    dyn_out = right_mid(BX_DYN)
    map_in  = (BX_MAP[0] + BX_MAP[2]*0.25, BX_MAP[1])  # bottom of mapper
    ortho_arrow(ax, dyn_out, map_in,
                via_x=BX_MAP[0] + BX_MAP[2]*0.25,
                color=ARROW_HOT, lw=1.3)
    label(ax, (dyn_out[0] + map_in[0])/2 + 0.20, dyn_out[1] + 0.04,
          "mask $M_t$", color=ARROW_HOT)

    # Dynamic -> Tracker (mask is also applied to the tracking loss; §3.2).
    # Short upward arrow within column 2. No label — same red colour and
    # same mask as the Mapper arrow already labelled "mask M_t". The
    # caption flags that both Tracker and Mapper losses use M_t.
    dyn_top = top_mid(BX_DYN)
    tracker_bot = bot_mid(BX_TRACK)
    arrow(ax, (dyn_top[0] - 0.55, dyn_top[1]),
              (tracker_bot[0] - 0.55, tracker_bot[1]),
          color=ARROW_HOT, lw=1.1)

    # Dynamic -> Gaussian Map (belief update — red dashed)
    dyn_bot = bot_mid(BX_DYN)
    gmap_top_left = (BX_GMAP[0] + BX_GMAP[2]*0.25, BX_GMAP[1] + BX_GMAP[3])
    ortho_arrow(ax, (dyn_bot[0], dyn_bot[1] - 0.02),
                gmap_top_left,
                via_y=4.10,
                color=ARROW_HOT, lw=1.0, ls="--")
    label(ax, 5.95, 4.07, "belief $b$ +=", color=ARROW_HOT)

    # Language extraction -> Autoencoder (CLIP feature)
    arrow(ax, bot_mid(BX_LANG), top_mid(BX_AE), lw=1.1)
    label(ax, bot_mid(BX_LANG)[0] + 0.7, (bot_mid(BX_LANG)[1] + top_mid(BX_AE)[1])/2,
          "$\\phi_{CLIP}\\!\\in\\!\\mathbb{R}^{768}$")

    # Autoencoder -> Mapper (lang supervision target).  Route up the
    # right side of column 2 so the label has space and we don't punch
    # through the Gaussian-Map title.
    ae_out = right_mid(BX_AE)
    via_lang_x = BX_MAP[0] - 0.30
    ortho_arrow(ax, ae_out, (BX_MAP[0], BX_MAP[1] + BX_MAP[3]*0.25),
                via_x=via_lang_x, color=ARROW_C, lw=1.1)
    label(ax, via_lang_x + 0.05, BX_AE[1] + BX_AE[3]/2 + 0.12,
          "$f_{\\!lang}\\!\\in\\!\\mathbb{R}^{16}$")

    # ----- Mapper <-> Gaussian Map (bidirectional) -----
    map_b = (BX_MAP[0] + BX_MAP[2]/2, BX_MAP[1])
    gmap_t = (BX_GMAP[0] + BX_GMAP[2]/2, BX_GMAP[1] + BX_GMAP[3])
    arrow(ax, map_b, gmap_t, color="#222", lw=1.4, style="<->")
    label(ax, map_b[0] + 0.50, (map_b[1] + gmap_t[1])/2,
          "grad / state")

    # ----- Gaussian Map -> Tracker (rendered RGB-D for photometric loss) -----
    # Dashed feedback loop: route up over the top of the Mapper into the
    # right edge of the Tracker. The tracker minimises L(I, hat I, D, hat D)
    # so it MUST see what the current map renders to from the prior pose.
    gmap_top_right = (BX_GMAP[0] + BX_GMAP[2]*0.85, BX_GMAP[1] + BX_GMAP[3])
    track_top_right = (BX_TRACK[0] + BX_TRACK[2] - 0.10,
                       BX_TRACK[1] + BX_TRACK[3])
    feedback_y = 5.78
    ax.plot([gmap_top_right[0], gmap_top_right[0],
             track_top_right[0], track_top_right[0]],
            [gmap_top_right[1], feedback_y,
             feedback_y, track_top_right[1] + 0.02],
            color="#3a6e9c", lw=1.0, ls="--", zorder=1,
            solid_capstyle="round")
    arrow(ax, (track_top_right[0], track_top_right[1] + 0.03),
              track_top_right, color="#3a6e9c", lw=1.0, ls="--")
    # Label sits just BELOW the horizontal feedback line, in the open gap
    # between the line (y=5.78) and the top of the Tracker / Mapper boxes
    # (y=5.50). Avoids collision with the column header at y=5.95.
    label(ax, (gmap_top_right[0] + track_top_right[0]) / 2,
              feedback_y - 0.13,
          "rendered $\\hat I,\\;\\hat D$ (feedback)", color="#3a6e9c")

    # ----- MAP -> OUTPUTS -----
    # All four right-side outputs share a vertical "rail" at out_rail_x;
    # each comes off the right edge of the Gaussian Map at its own y,
    # turns at the rail, and then arrows horizontally into the dest box.
    out_rail_x = BX_GMAP[0] + BX_GMAP[2] + 0.40

    # Trajectory (top output)
    traj_in = left_mid(BX_TRAJ)
    ortho_arrow(ax, right_mid(BX_GMAP), traj_in, via_x=out_rail_x)
    label(ax, out_rail_x + 0.15, traj_in[1] - 0.20, "poses")

    # Render
    rend_in = left_mid(BX_REND)
    rend_src = (BX_GMAP[0] + BX_GMAP[2], BX_GMAP[1] + BX_GMAP[3]*0.75)
    ortho_arrow(ax, rend_src, rend_in, via_x=out_rail_x, lw=1.1)
    label(ax, out_rail_x + 0.20, rend_in[1] - 0.18, "rasterize")

    # Cleaned map
    clean_in = left_mid(BX_CLEAN)
    clean_src = (BX_GMAP[0] + BX_GMAP[2], BX_GMAP[1] + BX_GMAP[3]*0.45)
    ortho_arrow(ax, clean_src, clean_in, via_x=out_rail_x, lw=1.1)
    label(ax, out_rail_x + 0.18, clean_in[1] + 0.18, "$b > \\theta$")

    # ----- QUERY STREAM (blue, bottom) -----
    # Text query -> Autoencoder (left side, low to mid). The query is first
    # CLIP-text-encoded with a 7-prompt ensemble (§3.5) into R^768, then
    # fed through the SAME scene autoencoder used to compress image features.
    arrow(ax, right_mid(BX_QUERY), left_mid(BX_AE),
          color=ARROW_QUERY, lw=1.2)
    label(ax, (right_mid(BX_QUERY)[0] + left_mid(BX_AE)[0]) / 2,
              (right_mid(BX_QUERY)[1] + left_mid(BX_AE)[1]) / 2 + 0.18,
          "CLIP-text $\\circ$ 7-prompt avg $\\to\\,\\mathbb{R}^{768}$",
          color=ARROW_QUERY)

    # Autoencoder -> Lang relevancy (route along the bottom of the figure)
    ae_query_out = (BX_AE[0] + BX_AE[2], BX_AE[1] + BX_AE[3]*0.30)
    rel_in = left_mid(BX_REL)
    via_y_q = 0.30
    ax.plot([ae_query_out[0], ae_query_out[0] + 0.40,
             ae_query_out[0] + 0.40, rel_in[0] - 0.40,
             rel_in[0] - 0.40, rel_in[0]],
            [ae_query_out[1], ae_query_out[1],
             via_y_q, via_y_q,
             rel_in[1], rel_in[1]],
            color=ARROW_QUERY, lw=1.2, zorder=1, solid_capstyle="round")
    arrow(ax, (rel_in[0] - 0.04, rel_in[1]), rel_in,
          color=ARROW_QUERY, lw=1.2)
    label(ax, (ae_query_out[0] + rel_in[0])/2, via_y_q + 0.18,
          "$q\\!\\in\\!\\mathbb{R}^{16}$  (query embedding)",
          color=ARROW_QUERY)

    # Gaussian Map taps the relevancy too: f vs q = relevancy.
    # Routed through the same vertical rail at out_rail_x as the other
    # Map-to-Output arrows for visual consistency.
    gmap_rel_src = (BX_GMAP[0] + BX_GMAP[2], BX_GMAP[1] + BX_GMAP[3]*0.18)
    ortho_arrow(ax, gmap_rel_src,
                (rel_in[0], rel_in[1] + 0.12),
                via_x=out_rail_x,
                color=ARROW_QUERY, lw=1.0, ls="--")

    # =====================================================================
    # Legend + title
    # =====================================================================
    legend_handles = [
        _swatch(CLR_INPUT,   "Input / query"),
        _swatch(CLR_TRACK,   "Geometric module"),
        _swatch(CLR_DYNAMIC, "Dynamic-object module"),
        _swatch(CLR_LANG,    "Language module"),
        _swatch(CLR_MAP,     "Map state"),
        _swatch(CLR_OUTPUT,  "Output"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.02),
              ncol=6, frameon=False, fontsize=8.5,
              handletextpad=0.4, columnspacing=1.3)

    fig.suptitle("DynLang-SLAM system architecture",
                 fontsize=12.5, fontweight="bold", y=0.995)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"figure_pipeline.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
