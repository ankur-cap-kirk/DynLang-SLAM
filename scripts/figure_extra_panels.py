"""Generate auxiliary panels for the pipeline diagram.

Produces:
    frame_060_depth_input.png   : colormapped depth from BONN H1 frame 60
    frame_060_clip_pca.png      : optional CLIP feature-map PCA visualisation
                                  (only generated if open_clip is importable)

These complete the "RGB-D" input pair and (optionally) the language-extraction
panel in figure_pipeline_v2.py without re-running SLAM.

Run:
    python scripts/figure_extra_panels.py
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

FIG_DIR  = os.path.join(PROJECT_ROOT, "results", "figures")
FILM_DIR = os.path.join(FIG_DIR, "filmstrip_h1")
os.makedirs(FILM_DIR, exist_ok=True)


# ── 1. Colormapped input depth at frame 60 ───────────────────────────────
def make_input_depth(frame_idx: int = 60, stride: int = 2,
                     n_frames: int = 100):
    """Load BONN H1 with same stride/N as filmstrip and dump a colormapped
    depth PNG for frame_idx."""
    from dynlang_slam.data.tum import TUMDataset

    ds = TUMDataset(
        data_dir=os.path.join(PROJECT_ROOT, "data", "BONN",
                              "rgbd_bonn_person_tracking"),
        height=480, width=640,
        depth_scale=5000.0,
        max_frames=n_frames,
        stride=stride,
    )
    sample = ds[frame_idx]
    # `depth` is metres after the dataset divides by depth_scale
    depth = sample["depth"].cpu().numpy()
    if depth.ndim == 3:
        depth = depth.squeeze()

    # Mask zeros (invalid pixels) so they don't dominate the colour range.
    valid = depth > 0
    if valid.any():
        d_min = float(depth[valid].min())
        d_max = float(np.percentile(depth[valid], 98))
    else:
        d_min, d_max = 0.0, 1.0

    norm = (depth - d_min) / max(d_max - d_min, 1e-6)
    norm = np.clip(norm, 0, 1)
    norm[~valid] = 0  # invalid pixels stay dark

    cmap = mpl.colormaps["turbo"]
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    # Black-out invalid pixels so the panel reads as "depth, missing→black"
    rgb[~valid] = (15, 15, 15)

    out_path = os.path.join(FILM_DIR, "frame_060_depth_input.png")
    Image.fromarray(rgb).save(out_path)
    print(f"Saved {out_path}  (depth range: {d_min:.2f}–{d_max:.2f} m)")


# ── 2. (Optional) CLIP feature-map PCA on frame 60 RGB ────────────────────
def make_clip_pca(frame_idx: int = 60, patch_grid: int = 30):
    """Run CLIP on the input RGB once and project per-patch features to 3D
    via PCA, normalised to RGB. Skipped silently if open_clip isn't
    available."""
    try:
        import torch
        import open_clip
    except Exception as e:
        print(f"[skip] open_clip not available: {e}")
        return

    rgb_path = os.path.join(FILM_DIR, "frame_060_rgb.png")
    if not os.path.exists(rgb_path):
        print(f"[skip] {rgb_path} not found")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip-pca] device={device}, loading CLIP-L/14 (this is slow first time)…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    model = model.to(device).eval()

    img = Image.open(rgb_path).convert("RGB")
    H, W = img.size[1], img.size[0]
    x = preprocess(img).unsqueeze(0).to(device)

    # Hook the visual transformer's penultimate token features
    feats_out = {}

    def _hook(module, inp, out):
        # out: (B, N_tokens, D) — keep patch tokens
        feats_out["t"] = out

    # OpenCLIP ViT-L/14 visual stack: model.visual
    vis = model.visual
    handle = vis.transformer.resblocks[-1].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model.encode_image(x)
    finally:
        handle.remove()
    tokens = feats_out["t"]    # (B, 1+N_patches, D)
    if tokens.dim() == 3 and tokens.shape[0] != 1:
        # OpenCLIP layouts can be (N+1, B, D); transpose if so
        tokens = tokens.permute(1, 0, 2)
    patches = tokens[0, 1:]    # drop CLS
    n = patches.shape[0]
    side = int(round(np.sqrt(n)))
    if side * side != n:
        print(f"[clip-pca] unexpected token count {n}")
        return
    feat = patches.float().cpu().numpy()        # (N, D)
    feat = feat - feat.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(feat, full_matrices=False)
    proj = U[:, :3] * S[:3]
    p_min = proj.min(axis=0)
    p_max = proj.max(axis=0)
    rgb_grid = (proj - p_min) / (p_max - p_min + 1e-6)
    rgb_grid = rgb_grid.reshape(side, side, 3)

    # Resize to original aspect for the diagram
    pca_img = Image.fromarray((rgb_grid * 255).astype(np.uint8))
    pca_img = pca_img.resize((W // 2, H // 2), Image.BILINEAR)

    out_path = os.path.join(FILM_DIR, "frame_060_clip_pca.png")
    pca_img.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_input_depth()
    # CLIP PCA is optional — comment out if you don't want it
    make_clip_pca()
