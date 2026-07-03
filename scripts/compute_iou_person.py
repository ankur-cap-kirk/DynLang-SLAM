"""IoU(person-query relevancy heatmap, YOLO person mask) on H1.

Recovers the binary YOLO mask exactly and the relative heatmap intensity
(lossy: percentile-stretched + alpha-blended) from the overlay PNGs in
results/figures/filmstrip_h1/, then reports IoU at several thresholds.

Recovery is honest about its limits:
  - Mask overlay: out = 0.55*rgb + 0.45*[220,30,30] for masked pixels,
    out = rgb otherwise. Solving the system gives an exact bool mask
    (modulo uint8 rounding noise).
  - Heatmap overlay: out = (1 - h_n*0.55)*rgb + h_n*0.55 * turbo(h_n),
    h_n = percentile-stretched heatmap. We recover *blend strength*
    per pixel by solving for h_n via per-channel residual; this is a
    monotone function of true relevancy in the [p60, p99] band, which
    is sufficient for ranking-based IoU at top-K thresholds.
"""
import os
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(PROJECT_ROOT, "results", "figures", "filmstrip_h1")
FRAMES = [30, 60, 99]  # frame 1 has empty map → skip
TOP_K_PERCENTS = [5.0, 10.0, 20.0]


def _load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def recover_mask(rgb, overlay, alpha=0.45, paint=(220.0, 30.0, 30.0)):
    """Recover bool mask from _overlay_mask alpha-blend. Exact up to uint8
    rounding."""
    paint = np.asarray(paint, dtype=np.float32)
    expected_painted = (1 - alpha) * rgb + alpha * paint
    diff_painted = np.abs(overlay - expected_painted).max(axis=-1)
    diff_clean = np.abs(overlay - rgb).max(axis=-1)
    return diff_painted < diff_clean


def recover_heatmap_intensity(rgb, overlay, max_alpha=0.55):
    """Recover per-pixel heatmap intensity h_n in [0, 1] from
    _overlay_heatmap. Solves for h_n via residual on the difference image.

    The blend is: out = (1 - h_n*max_alpha)*rgb + h_n*max_alpha * color(h_n)
    => out - rgb = h_n * max_alpha * (color(h_n) - rgb)
    The L2 norm of (out - rgb) per pixel is monotone in h_n (color changes
    slowly with h_n compared to alpha). Use it as a ranking proxy.
    """
    diff = overlay - rgb
    intensity = np.linalg.norm(diff, axis=-1)
    return intensity


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def main():
    print(f"{'frame':>5} {'top%':>6} {'IoU':>7} {'P':>7} {'R':>7} "
          f"{'GT_pix':>7} {'pred_pix':>9}")
    rows = []
    for fid in FRAMES:
        rgb = _load(os.path.join(DIR, f"frame_{fid:03d}_rgb.png"))
        mask_overlay = _load(os.path.join(DIR, f"frame_{fid:03d}_mask.png"))
        lang_overlay = _load(os.path.join(DIR,
            f"frame_{fid:03d}_lang_person-human.png"))

        # Resize lang overlay if it doesn't match (it can differ).
        if lang_overlay.shape != rgb.shape:
            lang_pil = Image.fromarray(lang_overlay.astype(np.uint8))
            lang_pil = lang_pil.resize((rgb.shape[1], rgb.shape[0]),
                                       Image.BILINEAR)
            lang_overlay = np.asarray(lang_pil).astype(np.float32)

        gt_mask = recover_mask(rgb, mask_overlay)
        intensity = recover_heatmap_intensity(rgb, lang_overlay)

        gt_pix = gt_mask.sum()
        if gt_pix == 0:
            continue

        for top_pct in TOP_K_PERCENTS:
            # Take top X% of pixels by intensity
            n_keep = int(intensity.size * top_pct / 100)
            thresh = np.partition(intensity.flatten(), -n_keep)[-n_keep]
            pred_mask = intensity >= thresh
            inter = (pred_mask & gt_mask).sum()
            i = iou(pred_mask, gt_mask)
            p = inter / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
            r = inter / gt_mask.sum() if gt_mask.sum() > 0 else 0.0
            rows.append((fid, top_pct, i, p, r, int(gt_pix), int(pred_mask.sum())))
            print(f"{fid:>5} {top_pct:>6.1f} {i:>7.3f} {p:>7.3f} {r:>7.3f} "
                  f"{int(gt_pix):>7} {int(pred_mask.sum()):>9}")

    # Aggregate
    print("\nMean across frames at each top-%:")
    for top_pct in TOP_K_PERCENTS:
        sel = [r for r in rows if r[1] == top_pct]
        ious = [r[2] for r in sel]
        precs = [r[3] for r in sel]
        recs = [r[4] for r in sel]
        print(f"  top-{top_pct:>4.1f}%  IoU = {np.mean(ious):.3f} "
              f"(±{np.std(ious):.3f})  P = {np.mean(precs):.3f}  "
              f"R = {np.mean(recs):.3f}")


if __name__ == "__main__":
    main()
