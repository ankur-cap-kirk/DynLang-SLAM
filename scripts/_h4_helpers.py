"""Shared helpers used by every scripts/h4_*.py figure script.

Kept private (underscore prefix) so it isn't picked up as a runnable test
script. Each helper here is intentionally small and side-effect-free so
the figure scripts can compose them without surprises.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynlang_slam.utils.config import load_config  # noqa: E402
from dynlang_slam.data.tum import (  # noqa: E402
    TUMDataset,
    load_intrinsics_from_file,
    get_pixel8_portrait_intrinsics,
)
from dynlang_slam.core.gaussians import GaussianMap  # noqa: E402
from dynlang_slam.slam.pipeline import SLAMPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@dataclass
class H4Setup:
    cfg: object
    dataset: TUMDataset
    intrinsics: dict
    gaussian_map: GaussianMap
    slam: SLAMPipeline
    device: str
    clip_dir: Path


def load_h4_setup(
    clip_dir: str | Path,
    n_frames: int,
    cfg_overrides: Optional[list[str]] = None,
    config_path: str = "configs/h4.yaml",
    device: str = "cuda",
    enable_dynamic: bool = True,
    enable_language: bool = True,
) -> H4Setup:
    """Build a SLAM pipeline ready to run on an H4 clip directory.

    Caller is responsible for actually running ``slam.process_first_frame``
    and the per-frame loop. We just load + wire everything.
    """
    clip_dir = Path(clip_dir).resolve()
    if not clip_dir.exists():
        raise SystemExit(f"clip dir not found: {clip_dir}")

    cfg = load_config(config_path, cfg_overrides or [])
    cfg.dynamic.enabled = enable_dynamic
    cfg.language.enabled = enable_language
    cfg.dataset.has_gt = False
    cfg.dataset.max_frames = n_frames

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    intrinsics_file = clip_dir / "intrinsics.txt"
    h, w = cfg.dataset.image_height, cfg.dataset.image_width
    if intrinsics_file.exists():
        intrinsics = load_intrinsics_from_file(str(intrinsics_file), h, w)
    else:
        intrinsics = get_pixel8_portrait_intrinsics(h, w)

    dataset = TUMDataset(
        data_dir=str(clip_dir),
        height=h,
        width=w,
        depth_scale=cfg.dataset.depth_scale,
        max_frames=n_frames,
        has_gt=False,
    )

    gaussian_map = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
    return H4Setup(
        cfg=cfg, dataset=dataset, intrinsics=intrinsics,
        gaussian_map=gaussian_map, slam=slam, device=device,
        clip_dir=clip_dir,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_at_pose(
    slam: SLAMPipeline,
    gaussian_map: GaussianMap,
    viewmat: torch.Tensor,
    width: int,
    height: int,
    render_lang: bool = False,
) -> dict:
    """Render the current map from a given world-to-camera viewmat."""
    return slam.renderer(
        gaussian_map=gaussian_map,
        viewmat=viewmat.to(slam.device),
        K=slam.K,
        width=width,
        height=height,
        render_lang=render_lang,
    )


def to_uint8_rgb(rgb_tensor: torch.Tensor) -> np.ndarray:
    """(H, W, 3) float [0,1] -> (H, W, 3) uint8."""
    arr = rgb_tensor.detach().cpu().numpy()
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def frame_rgb_to_uint8(frame: dict) -> np.ndarray:
    """Dataset frame['rgb'] (3, H, W) float [0,1] -> (H, W, 3) uint8."""
    rgb = frame["rgb"].permute(1, 2, 0).cpu().numpy()
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask painting
# ---------------------------------------------------------------------------

def paint_mask(
    rgb_uint8: np.ndarray,
    mask_bool: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a binary mask onto an RGB image. mask is (H, W) bool/float."""
    out = rgb_uint8.astype(np.float32).copy()
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool > 0.5
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask_bool] = (
        out[mask_bool] * (1.0 - alpha) + color_arr * alpha
    ).reshape(-1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_turbo(
    scalar: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map a (H, W) scalar field to a (H, W, 3) uint8 turbo colormap."""
    import matplotlib.cm as cm
    if vmin is None:
        vmin = float(np.nanmin(scalar))
    if vmax is None:
        vmax = float(np.nanmax(scalar))
    if vmax - vmin < 1e-6:
        norm = np.zeros_like(scalar, dtype=np.float32)
    else:
        norm = np.clip((scalar - vmin) / (vmax - vmin), 0.0, 1.0)
    colored = cm.turbo(norm)[:, :, :3]
    return (colored * 255.0).astype(np.uint8)


def overlay_heatmap(
    rgb_uint8: np.ndarray,
    heat_uint8: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    return np.clip(
        rgb_uint8.astype(np.float32) * (1.0 - alpha)
        + heat_uint8.astype(np.float32) * alpha,
        0, 255,
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-Gaussian scalar field rasterizer (for belief, contamination, etc.)
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_per_gaussian_scalar(
    gaussian_map: GaussianMap,
    scalar: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a per-Gaussian scalar field through gsplat.

    Returns (H, W) scalar image and (H, W) alpha image. The returned scalar
    is alpha-premultiplied; divide by alpha (where alpha > eps) to get the
    weighted-mean scalar per pixel, or leave as-is for a "support intensity"
    visualization.
    """
    from gsplat import rasterization
    if gaussian_map.num_gaussians == 0 or scalar.numel() == 0:
        return (np.zeros((height, width), dtype=np.float32),
                np.zeros((height, width), dtype=np.float32))

    params = gaussian_map.get_activated_params()
    means = params["means"]
    quats = params["quats"]
    scales = params["scales"]
    opacities = params["opacities"].squeeze(-1)
    if scalar.dim() == 1:
        colors = scalar.unsqueeze(-1).expand(-1, 3).contiguous()
    else:
        colors = scalar
    rendered, alphas, _ = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width, height=height,
        near_plane=near, far_plane=far,
        render_mode="RGB",
        packed=True, sh_degree=None,
    )
    scalar_img = rendered[0, :, :, 0].detach().cpu().numpy()  # all 3 channels equal
    alpha_img = alphas[0, :, :, 0].detach().cpu().numpy()
    return scalar_img, alpha_img


# ---------------------------------------------------------------------------
# Relevancy on the rendered language map
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_relevancy(
    slam: SLAMPipeline,
    gaussian_map: GaussianMap,
    viewmat: torch.Tensor,
    width: int,
    height: int,
    queries: list[str],
    canonical_phrases: tuple[str, ...] = ("object", "things", "stuff", "texture"),
    temperature: float = 50.0,
) -> dict[str, np.ndarray]:
    """Return {query: (H, W) relevancy map in [0,1]} for several queries.

    Cheap because we render the language feature map once and reuse it.
    """
    if not slam._lang_initialized or slam._autoencoder is None:
        raise RuntimeError("Language pipeline not initialized; "
                           "run SLAM with language enabled and at least"
                           " enough keyframes for AE warmup to finish.")

    out = render_at_pose(
        slam, gaussian_map, viewmat,
        width=width, height=height, render_lang=True,
    )
    if "lang_feats" not in out:
        raise RuntimeError("renderer returned no lang_feats; "
                           "is gaussian_map.lang_feats populated?")
    feat_map = torch.nn.functional.normalize(out["lang_feats"], dim=-1)  # (H, W, D)

    # Bring CLIP onto GPU briefly to encode queries + canonical phrases
    slam._clip_extractor.model.to(slam.device)
    slam._clip_extractor.device = slam.device
    try:
        canon_clip = slam._clip_extractor.encode_texts(list(canonical_phrases)).to(slam.device)
        canon_lat = slam._autoencoder.encode(canon_clip)
        canon_lat = torch.nn.functional.normalize(canon_lat, dim=-1)  # (C, D)
        sim_canon_per_pixel = torch.einsum("hwd,cd->hwc", feat_map, canon_lat)
        max_canon = sim_canon_per_pixel.max(dim=-1).values  # (H, W)

        results: dict[str, np.ndarray] = {}
        for q in queries:
            q_clip = slam._clip_extractor.encode_text(q).to(slam.device)
            q_lat = slam._autoencoder.encode(q_clip.unsqueeze(0)).squeeze(0)
            q_lat = torch.nn.functional.normalize(q_lat, dim=-1)
            sim_q = torch.einsum("hwd,d->hw", feat_map, q_lat)
            num = torch.exp(temperature * sim_q)
            den = num + torch.exp(temperature * max_canon)
            rel = (num / den).cpu().numpy()
            results[q] = rel
    finally:
        slam._clip_extractor.model.to("cpu")
        slam._clip_extractor.device = "cpu"
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_grid(
    panels: list[list[np.ndarray]],
    out_path: Path,
    titles_top: Optional[list[str]] = None,
    row_labels: Optional[list[str]] = None,
    border_px: int = 6,
    bg_color: int = 255,
) -> None:
    """Compose a 2D grid of RGB uint8 panels and save it.

    All panels in a column must share the same width. All panels in a row
    must share the same height. Differences are padded.
    """
    if not panels or not panels[0]:
        raise ValueError("panels must be non-empty")

    rows = len(panels)
    cols = len(panels[0])
    for r, row in enumerate(panels):
        if len(row) != cols:
            raise ValueError(
                f"row {r} has {len(row)} panels, expected {cols}"
            )

    col_widths = [max(panels[r][c].shape[1] for r in range(rows)) for c in range(cols)]
    row_heights = [max(panels[r][c].shape[0] for c in range(cols)) for r in range(rows)]

    label_pad = 0
    if row_labels is not None or titles_top is not None:
        label_pad = 22  # text band
    title_band = label_pad if titles_top is not None else 0
    side_band = 60 if row_labels is not None else 0

    total_w = side_band + sum(col_widths) + (cols + 1) * border_px
    total_h = title_band + sum(row_heights) + (rows + 1) * border_px

    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    if titles_top is not None:
        if len(titles_top) != cols:
            raise ValueError(f"titles_top has {len(titles_top)} entries, expected {cols}")
        for c, title in enumerate(titles_top):
            x0 = side_band + border_px + sum(col_widths[:c]) + c * border_px
            cv2.putText(canvas, title,
                        (x0 + 4, title_band - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    for r in range(rows):
        for c in range(cols):
            panel = panels[r][c]
            x0 = side_band + border_px + sum(col_widths[:c]) + c * border_px
            y0 = title_band + border_px + sum(row_heights[:r]) + r * border_px
            ph, pw = panel.shape[:2]
            canvas[y0:y0 + ph, x0:x0 + pw] = panel
        if row_labels is not None:
            r_label = row_labels[r]
            y_text = title_band + border_px + sum(row_heights[:r]) + r * border_px + 16
            cv2.putText(canvas, r_label,
                        (4, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
