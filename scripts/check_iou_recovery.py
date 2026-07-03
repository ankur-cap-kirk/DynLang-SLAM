"""Visual sanity check on the heatmap/mask recovery used by
compute_iou_person.py. Saves a 4-panel comparison PNG."""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(PROJECT_ROOT, "results", "figures", "filmstrip_h1")
FRAME = 99
OUT = os.path.join(PROJECT_ROOT, "results", "figures",
                   f"_iou_recovery_check_f{FRAME:03d}.png")


def _load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def recover_mask(rgb, overlay, alpha=0.45, paint=(220.0, 30.0, 30.0)):
    paint = np.asarray(paint, dtype=np.float32)
    expected_painted = (1 - alpha) * rgb + alpha * paint
    diff_painted = np.abs(overlay - expected_painted).max(axis=-1)
    diff_clean = np.abs(overlay - rgb).max(axis=-1)
    return diff_painted < diff_clean


def recover_heatmap_intensity(rgb, overlay):
    diff = overlay - rgb
    return np.linalg.norm(diff, axis=-1)


rgb = _load(os.path.join(DIR, f"frame_{FRAME:03d}_rgb.png"))
mask_overlay = _load(os.path.join(DIR, f"frame_{FRAME:03d}_mask.png"))
lang_overlay = _load(os.path.join(DIR,
    f"frame_{FRAME:03d}_lang_person-human.png"))

gt_mask = recover_mask(rgb, mask_overlay)
intensity = recover_heatmap_intensity(rgb, lang_overlay)

# Build top-10% predicted mask
n_keep = int(intensity.size * 0.10)
thresh = np.partition(intensity.flatten(), -n_keep)[-n_keep]
pred_mask = intensity >= thresh

fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
axs[0,0].imshow(rgb.astype(np.uint8))
axs[0,0].set_title(f"input RGB (frame {FRAME})")
axs[0,1].imshow(lang_overlay.astype(np.uint8))
axs[0,1].set_title("lang_person-human overlay (provided)")
axs[1,0].imshow(rgb.astype(np.uint8))
axs[1,0].imshow(gt_mask, cmap="Reds", alpha=0.45)
axs[1,0].set_title(f"recovered YOLO mask "
                   f"({100*gt_mask.mean():.1f}% pixels)")
axs[1,1].imshow(rgb.astype(np.uint8))
axs[1,1].imshow(pred_mask, cmap="Greens", alpha=0.45)
inter = (pred_mask & gt_mask).sum()
union = (pred_mask | gt_mask).sum()
iou = inter / union if union else 0
axs[1,1].set_title(f"top-10% recovered heat (IoU={iou:.3f})")
for a in axs.flatten():
    a.set_xticks([]); a.set_yticks([])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
