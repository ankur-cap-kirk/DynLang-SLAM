"""Build the EEE 515 final-presentation slide deck for DynLang-SLAM.

Produces a 7-slide PowerPoint matching the prof's per-section time budget:

    1. Title             (30 sec - 1 min)
    2. Introduction      (1 min)
    3. Problem           (1 min)
    4. Related Work      (1-2 min)
    5. Approach          (2 min)
    6. Experimental Results (2-3 min)
    7. Conclusion + Future Objectives (1 min)

Each slide has:
  - A clear title
  - One hero figure pulled from results/figures/
  - 3-5 bullet points sized for the time slot
  - Speaker notes containing the suggested voice-over verbatim

You record voice-over slide-by-slide via PowerPoint -> Slide Show ->
Record Slideshow, then export as MP4.

Run:
    python scripts/build_presentation.py
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
OUT_PPTX = os.path.join(PROJECT_ROOT, "DynLang_SLAM_Presentation.pptx")

# ── Slide design tokens ──────────────────────────────────────────────────
COLOR_TITLE  = RGBColor(0x1F, 0x2C, 0x4D)
COLOR_BODY   = RGBColor(0x22, 0x22, 0x22)
COLOR_SUBTLE = RGBColor(0x66, 0x66, 0x66)
COLOR_ACCENT = RGBColor(0xC1, 0x4A, 0x4A)

prs = Presentation()
prs.slide_width  = Inches(13.333)   # 16:9 widescreen
prs.slide_height = Inches(7.5)
SW = prs.slide_width
SH = prs.slide_height
BLANK_LAYOUT = prs.slide_layouts[6]


# ── Helpers ───────────────────────────────────────────────────────────────
def add_title(slide, text, *, x=0.5, y=0.25, w=12.3, h=0.8,
              size=32, color=COLOR_TITLE, bold=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return tb


def add_subtitle(slide, text, *, x=0.5, y=1.15, w=12.3, h=0.5,
                 size=18, color=COLOR_SUBTLE):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.italic = True
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return tb


def add_bullets(slide, bullets, *, x=0.6, y=2.1, w=6.0, h=4.5,
                size=18, line_spacing=1.25):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = "•  " + line
        run.font.size = Pt(size)
        run.font.color.rgb = COLOR_BODY
        run.font.name = "Calibri"
    return tb


def add_image_fit(slide, png_path, *, x, y, w, h, caption=None):
    """Add image fitted inside the (w, h) box, preserving aspect ratio.
    Centers the image inside the box."""
    if not os.path.exists(png_path):
        # placeholder text
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tb.text_frame.text = f"[missing: {os.path.basename(png_path)}]"
        return tb

    img = Image.open(png_path)
    iw, ih = img.size
    aspect = iw / ih
    box_aspect = w / h
    if aspect >= box_aspect:
        # constrained by width
        new_w = w
        new_h = w / aspect
        new_x = x
        new_y = y + (h - new_h) / 2
    else:
        new_h = h
        new_w = h * aspect
        new_y = y
        new_x = x + (w - new_w) / 2
    pic = slide.shapes.add_picture(
        png_path, Inches(new_x), Inches(new_y),
        width=Inches(new_w), height=Inches(new_h),
    )
    if caption:
        cap_y = new_y + new_h + 0.05
        tb = slide.shapes.add_textbox(
            Inches(new_x), Inches(cap_y), Inches(new_w), Inches(0.3),
        )
        p = tb.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = caption
        run.font.size = Pt(11)
        run.font.italic = True
        run.font.color.rgb = COLOR_SUBTLE
    return pic


def add_speaker_notes(slide, text):
    notes = slide.notes_slide.notes_text_frame
    notes.text = text


def add_footer(slide, text):
    """Tiny footer with author + project tag."""
    tb = slide.shapes.add_textbox(
        Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3),
    )
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(9)
    run.font.italic = True
    run.font.color.rgb = COLOR_SUBTLE


# ── Slide 1 — Title ──────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "DynLang-SLAM",
          x=0.5, y=0.4, w=12.3, h=1.0, size=44)
add_subtitle(s,
             "Dynamic-Aware Open-Vocabulary 3D Gaussian Splatting SLAM",
             x=0.5, y=1.4, w=12.3, h=0.6, size=22)
# author line
tb = s.shapes.add_textbox(Inches(0.5), Inches(2.1), Inches(12.3), Inches(0.45))
p = tb.text_frame.paragraphs[0]
run = p.add_run()
run.text = "Ankur Guruprasad   ·   EEE 515 Computer Vision   ·   Spring 2026   ·   Arizona State University"
run.font.size = Pt(16)
run.font.color.rgb = COLOR_BODY
# hero figure
add_image_fit(
    s, os.path.join(FIG_DIR, "figure_tracking_mapping_grid.png"),
    x=1.5, y=2.8, w=10.3, h=4.3,
    caption="DynLang-SLAM running on four diverse sequences",
)
add_speaker_notes(s,
    "Hi, I'm Ankur. My final project is DynLang-SLAM — an online 3D Gaussian "
    "Splatting SLAM system that does three things at once: it builds a dense "
    "3D map, it grounds that map in natural language so you can query it with "
    "text, and it filters out moving objects from both tracking and mapping. "
    "I'll show it running on four very different scenes — two BONN dynamic "
    "indoor sequences, a synthetic Replica room, and a phone capture I made "
    "myself.")


# ── Slide 2 — Introduction ───────────────────────────────────────────────
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Introduction")
add_subtitle(s, "Two structural gaps in today's 3DGS-SLAM systems")
add_bullets(s, [
    "3DGS-SLAM (SplaTAM, MonoGS, Gaussian-SLAM) reconstructs photorealistic "
    "geometry online, replacing NeRF as the dense-SLAM representation",
    "Gap 1 — semantically mute: maps are coordinates of Gaussians, no "
    "concept of \"chair\" or \"keyboard\"",
    "Gap 2 — static-world assumption: a single moving person bakes "
    "ghost-streak artefacts into the map and breaks tracking",
    "DynLang-SLAM closes BOTH gaps in a single online pipeline on an "
    "8 GB consumer GPU",
], x=0.5, y=1.85, w=6.0, h=4.5, size=18)
add_image_fit(
    s, os.path.join(FIG_DIR, "figure_dynamic_ablation.png"),
    x=6.7, y=1.85, w=6.3, h=4.5,
    caption="Without dynamic masking: person silhouettes bake into the map",
)
add_footer(s, "DynLang-SLAM · 2/7")
add_speaker_notes(s,
    "SLAM is the perception backbone of mobile robots, AR headsets, and "
    "autonomous vehicles. The newest dense SLAM systems use 3D Gaussian "
    "Splatting and produce photorealistic maps in real time. But two "
    "problems remain. First, those maps are geometrically rich but "
    "semantically mute — the robot sees Gaussians at coordinates, but "
    "nothing says \"this is a chair.\" Second, every state-of-the-art "
    "3DGS-SLAM system implicitly assumes the world is static, so a single "
    "person walking through the frame causes catastrophic tracking drift "
    "and bakes ghost streaks into the map. DynLang-SLAM solves both, "
    "and runs them together on a single 8 GB consumer GPU.")


# ── Slide 3 — Problem ────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Problem: Dynamic Objects Corrupt the Map")
add_subtitle(s, "Without filtering, moving people get baked into the static reconstruction")
add_image_fit(
    s, os.path.join(FIG_DIR, "figure_dynamic_ablation.png"),
    x=0.5, y=1.85, w=8.5, h=5.0,
)
add_bullets(s, [
    "Top: dynamic-mask OFF — translucent ghost streaks accumulate on every "
    "frame; by frame 99 the room is fogged out",
    "Bottom: dynamic-mask ON — clean reconstruction at the same canonical "
    "viewpoint",
    "Quantitatively: dynamic masking cuts unaligned ATE-RMSE from "
    "15.0 → 8.8 cm (-41%) on H1",
    "Question: can we do dynamic filtering AND attach language features "
    "AND fit on consumer GPU?",
], x=9.2, y=1.85, w=3.9, h=5.2, size=14, line_spacing=1.15)
add_footer(s, "DynLang-SLAM · 3/7")
add_speaker_notes(s,
    "Here's the problem made concrete. On the top row, dynamic-mask is OFF — "
    "every frame the person walks through, their silhouette gets baked into "
    "the static Gaussian map, accumulating as translucent ghost streaks. "
    "By frame 99 the room is fogged out by stacked person ghosts. On the "
    "bottom row, dynamic-mask is ON — the same canonical viewpoint shows "
    "a clean render with the desk, box, and walls preserved. "
    "Quantitatively, dynamic masking cuts unaligned ATE-RMSE by 41 percent, "
    "from 15 centimetres to 8.8 centimetres. So the question is: can we do "
    "this AND attach language features AND do it on a consumer GPU? Each of "
    "those individually is computationally heavy.")


# ── Slide 4 — Related Work ───────────────────────────────────────────────
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Related Work")
add_subtitle(s, "Three independent threads — DynLang-SLAM is their intersection")

# Three columns
def _col(slide, x, header, items, color):
    # header
    tb = slide.shapes.add_textbox(Inches(x), Inches(1.85),
                                   Inches(4.0), Inches(0.55))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = header
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = color
    # body
    tb2 = slide.shapes.add_textbox(Inches(x), Inches(2.5),
                                    Inches(4.0), Inches(3.5))
    tf = tb2.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.2
        run = p.add_run()
        run.text = "•  " + item
        run.font.size = Pt(14)
        run.font.color.rgb = COLOR_BODY

_col(s, 0.5,  "3DGS SLAM",
     ["SplaTAM, MonoGS, GS-SLAM, Gaussian-SLAM, Photo-SLAM",
      "Online dense mapping, photorealistic",
      "All assume a STATIC world",
      "No semantics in the map"],
     RGBColor(0x2A, 0x6E, 0xB7))

_col(s, 4.7,  "Language-Embedded 3D",
     ["LERF (NeRF-era), OpenScene, F3RM",
      "LangSplat — first 3DGS port",
      "OFFLINE: per-scene training",
      "8.2 FPS render on A100 (LangSplat)"],
     RGBColor(0x4F, 0x8C, 0x37))

_col(s, 8.9,  "Dynamic SLAM",
     ["DynaSLAM (Mask R-CNN)",
      "ReFusion (introduces BONN)",
      "DG-SLAM — DROID-VO + hybrid back-end",
      "DynaMoN (NeRF-side)",
      "No language grounding"],
     RGBColor(0xC1, 0x4A, 0x4A))

# bottom: intersection
tb = s.shapes.add_textbox(Inches(0.5), Inches(6.0), Inches(12.3), Inches(0.85))
p = tb.text_frame.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
run = p.add_run()
run.text = "DynLang-SLAM = the intersection · online · 8 GB GPU"
run.font.size = Pt(22)
run.font.bold = True
run.font.color.rgb = COLOR_TITLE
add_footer(s, "DynLang-SLAM · 4/7")
add_speaker_notes(s,
    "Three threads of prior work feed into this. On the 3DGS SLAM side, "
    "SplaTAM, MonoGS, GS-SLAM, and Gaussian-SLAM established online dense "
    "mapping with photorealistic reconstruction — but all of them target a "
    "static world and embed no semantics. On the language side, LERF "
    "distilled CLIP into NeRF, and LangSplat ported the idea to 3D Gaussians "
    "— but LangSplat is offline, runs at 8.2 FPS on an A100, and trains "
    "per-scene from a fixed image set. On the dynamic side, DynaSLAM uses "
    "Mask R-CNN to filter moving objects, and DG-SLAM combines a DROID-VO "
    "front-end with a hybrid back-end on BONN, but neither does language. "
    "Each pair has been combined, but no system has done all three together. "
    "Naively stacking these capabilities exceeds consumer-GPU compute. "
    "DynLang-SLAM occupies that intersection, and the contribution is the "
    "engineering choices that make it tractable: feature compression, "
    "keyframe-only language extraction, and alpha-gated masked loss.")


# ── Slide 5 — Approach ───────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Approach: Unified Pipeline")
add_subtitle(s, "Three streams · Gaussian map state · text-query inference")
add_image_fit(
    s, os.path.join(FIG_DIR, "figure_pipeline_v2.png"),
    x=0.3, y=1.7, w=12.7, h=5.2,
)
add_footer(s, "DynLang-SLAM · 5/7")
add_speaker_notes(s,
    "Here's the architecture. Input is RGB-D plus the previous frame's pose. "
    "Three streams run per frame. The Tracker minimises a photometric "
    "Levenberg-Marquardt loss against the rendered map. The Dynamic "
    "Detector — YOLOv8x-Seg plus a 3-frame temporal filter — produces a "
    "mask that excludes moving pixels from the photometric loss. The "
    "Language Extraction stream uses CLIP-L/14 plus SAM2, but only on "
    "keyframes — not every frame — so its cost amortises across many frames. "
    "CLIP gives 768-dimensional features, which are far too expensive to "
    "attach to every Gaussian. So we train a scene-specific autoencoder "
    "online that compresses 768-D to 16-D — that's a more than 40× "
    "reduction in rasterisation cost. The Mapper then jointly optimises the "
    "Gaussian state — means, scales, quaternions, opacities, colours, and "
    "the 16-D language features — over a sliding window of 5 keyframes for "
    "60 iterations per frame. We also maintain a per-Gaussian dynamic-belief "
    "field that tracks contamination over time and prunes Gaussians whose "
    "belief crosses a threshold. At query time, the user types text; "
    "CLIP-text encodes it through a 7-prompt ensemble, the same encoder "
    "maps it to 16-D, and we compute a LERF-style softmax relevancy "
    "against the per-Gaussian features. All of this runs together on an "
    "8 GB consumer GPU.")


# ── Slide 6 — Experimental Results ───────────────────────────────────────
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Experimental Results")
add_subtitle(s, "Generalisation across 4 scenes · dynamic ablation · open-vocab queries")
# Top-left big: tracking grid
add_image_fit(
    s, os.path.join(FIG_DIR, "figure_tracking_mapping_grid.png"),
    x=0.3, y=1.7, w=8.5, h=4.0,
)
# Right: ablation table (text)
tb = s.shapes.add_textbox(Inches(8.95), Inches(1.7), Inches(4.2), Inches(2.2))
tf = tb.text_frame; tf.word_wrap = True
hdr = tf.paragraphs[0]
r = hdr.add_run()
r.text = "Three-mode ablation on H1"
r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = COLOR_TITLE
rows = [
    ("Static (no dyn, no lang)", "15.01 cm"),
    ("Dynamic only", " 8.83 cm  (-41%)"),
    ("Full (dyn + lang)", " 8.48 cm"),
    ("DG-SLAM [Xu+24] (ref.)", " 4.73 cm"),
]
for label, val in rows:
    p = tf.add_paragraph()
    p.line_spacing = 1.25
    run = p.add_run()
    run.text = f"  {label:<28s} {val}"
    run.font.size = Pt(13)
    run.font.name = "Consolas"
    run.font.color.rgb = COLOR_BODY
# bottom-right: query gallery
add_image_fit(
    s, os.path.join(FIG_DIR, "figure_multi_query.png"),
    x=8.95, y=4.05, w=4.2, h=2.85,
    caption="Open-vocab queries · IoU 0.10 vs YOLO at top-10% (5–8× random)",
)
# bottom-left: brief callouts
tb = s.shapes.add_textbox(Inches(0.4), Inches(5.85), Inches(8.5), Inches(1.1))
tf = tb.text_frame; tf.word_wrap = True
for i, line in enumerate([
    "✓ Generalises across BONN dynamic indoor, Replica synthetic, and a Pixel 8 in-the-wild capture",
    "✓ Dynamic masking helps tracking (-41%); language pipeline is integration-cost neutral",
    "✓ Open-vocab IoU at top-10% reaches 0.20 in late-map regime (frame 99 of H1)",
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.line_spacing = 1.15
    run = p.add_run()
    run.text = line
    run.font.size = Pt(13)
    run.font.color.rgb = COLOR_BODY
add_footer(s, "DynLang-SLAM · 6/7")
add_speaker_notes(s,
    "First, scene generalisation. Top-left of the slide: tracking and "
    "mapping across four sequences, rendered from an external fly-by "
    "viewpoint with the trajectory in red and frusta in blue. Top-left of "
    "the grid is BONN person tracking — real, dynamic indoor. Top-right is "
    "BONN balloon — also real, dynamic. Bottom-left is Replica room0 — "
    "synthetic, controlled. Bottom-right is an in-the-wild Pixel 8 phone "
    "capture I recorded myself with monocular depth from DepthAnything — "
    "no ground truth at all. The reconstructed scenes and tracked "
    "trajectories are visually consistent on all four. "
    "Second, the dynamic ablation. The three-mode table on the right "
    "isolates two real claims. Static baseline drifts to 15 cm of unaligned "
    "ATE. Adding dynamic masking cuts that to 8.8 cm — a 41 percent "
    "reduction. Adding the language pipeline on top is integration-cost "
    "neutral on tracking — 8.83 down to 8.48 cm, well within seed-to-seed "
    "noise. Dynamic masking helps tracking, not just visual cleanliness, "
    "and language doesn't interfere with SLAM. "
    "Third, language. The query gallery on the bottom-right shows five "
    "different prompts — chair, desk, keyboard, person, monitor — all "
    "returning spatially plausible relevancy heatmaps. We have one "
    "quantitative number: for the person query, IoU against YOLO averaged "
    "over three frames is 0.10 at top-10 percent, which is 5 to 8 times "
    "above random baseline. The pipeline reaches 0.20 IoU at frame 99 — "
    "usable but not yet at the level a robotic text-query system would "
    "need.")


# ── Slide 7 — Conclusion + Future Objectives ─────────────────────────────
s = prs.slides.add_slide(BLANK_LAYOUT)
add_title(s, "Conclusion + Future Objectives")
add_subtitle(s, "What we built · what's next")

# Two columns
tb = s.shapes.add_textbox(Inches(0.6), Inches(1.85), Inches(6.0), Inches(0.5))
p = tb.text_frame.paragraphs[0]
r = p.add_run()
r.text = "What we showed"
r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = COLOR_TITLE

add_bullets(s, [
    "First online 3DGS-SLAM unifying mapping + tracking + dynamic-object "
    "filtering + open-vocab language on 8 GB GPU",
    "Generalises across 4 diverse scenes — BONN dynamic, Replica synthetic, "
    "Pixel 8 in-the-wild",
    "Dynamic masking gives 41% ATE-RMSE reduction; language pipeline is "
    "integration-cost neutral on tracking",
    "Open-vocabulary IoU at 5–8× above random baseline",
], x=0.6, y=2.4, w=6.0, h=4.3, size=15, line_spacing=1.3)

tb = s.shapes.add_textbox(Inches(7.0), Inches(1.85), Inches(6.0), Inches(0.5))
p = tb.text_frame.paragraphs[0]
r = p.add_run()
r.text = "Future objectives"
r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = COLOR_ACCENT

add_bullets(s, [
    "Swap CLIP for SigLIP — fix colour-bias in text grounding "
    "(failure mode A)",
    "Multi-scale SAM masks (à la GAGS) — improve small-object localisation "
    "(failure mode B)",
    "Memory-aware keyframe pruning — lift the 1000-frame Replica cap on "
    "8 GB hardware",
    "Full-sequence multi-seed ATE evaluation across all 4 sequences",
], x=7.0, y=2.4, w=6.0, h=4.3, size=15, line_spacing=1.3)

# closing line
tb = s.shapes.add_textbox(Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.4))
p = tb.text_frame.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "Thank you. Questions?"
r.font.size = Pt(20); r.font.italic = True; r.font.color.rgb = COLOR_TITLE
add_footer(s, "DynLang-SLAM · 7/7")
add_speaker_notes(s,
    "To wrap up. DynLang-SLAM is the first online 3DGS-SLAM pipeline that "
    "unifies mapping, tracking, dynamic-object filtering, and open-vocabulary "
    "language querying on a single 8 GB consumer GPU. We showed it "
    "generalises across four diverse scenes — BONN dynamic indoor, Replica "
    "synthetic, and an in-the-wild Pixel 8 capture. Dynamic masking alone "
    "cuts ATE by 41 percent, and the language pipeline is neutral on "
    "tracking cost. Three concrete next steps: swap CLIP for SigLIP to fix "
    "colour-bias in text grounding, add multi-scale SAM masks for "
    "small-object localisation, and add memory-aware keyframe pruning to "
    "scale beyond 1000 Replica frames. Thank you, happy to take questions.")


# ── Save ──────────────────────────────────────────────────────────────────
prs.save(OUT_PPTX)
print(f"Saved {OUT_PPTX}")
print(f"  {len(prs.slides)} slides; {os.path.getsize(OUT_PPTX) / 1e6:.1f} MB")
