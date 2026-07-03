"""Qualitative filmstrip figure: tracking + mapping + language in action over time.

Runs DynLang-SLAM on a chosen scene and captures, at K evenly spaced
frames, four per-frame visualizations:

    Row 1 (Input):    the RGB frame fed to the pipeline.
    Row 2 (Render):   the current Gaussian map rasterized from the
                      just-tracked camera pose. Demonstrates that the map
                      reproduces the input -- tracking + mapping coupled.
    Row 3 (Mask):     the dynamic-object mask (YOLO) overlaid in
                      semi-transparent red on the input.
    Row 4 (Language): per-pixel relevancy heatmap for a text query
                      (e.g. "person", "table"), rendered from the
                      Gaussian map's language features. Hot = high
                      relevance to the query.

Composites the captured frames into a single PDF/PNG with row labels
on the left and frame indices on top -- the standard SLAM-paper
qualitative figure. Per-frame PNGs are also saved.

Usage:
    python scripts/filmstrip.py                          # H1, query="person"
    DYNLANG_SCENE=h3 DYNLANG_QUERY=chair python scripts/filmstrip.py
    DYNLANG_FRAMES=1,20,40,60,80,99 python scripts/filmstrip.py
    DYNLANG_LANG=0 python scripts/filmstrip.py           # 3-row version (faster)
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

from dynlang_slam.utils.config import load_config
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline
from dynlang_slam.utils.camera import fast_se3_inverse


# ---- Config -------------------------------------------------------------
SCENE = os.environ.get("DYNLANG_SCENE", "h1").lower()
SEED = int(os.environ.get("DYNLANG_SEED", "42"))
N_BONN = int(os.environ.get("DYNLANG_N_FRAMES_BONN", "100"))
STRIDE_BONN = int(os.environ.get("DYNLANG_STRIDE_BONN", "2"))
N_REPLICA = int(os.environ.get("DYNLANG_N_FRAMES_REPLICA", "100"))

# Default text queries per scene. Picked so the heatmap LIGHTS UP --
# i.e. matches a STATIC object that survived dynamic masking. Querying
# "person" on H1 would (correctly) yield a cold heatmap because the
# dynamic pipeline filters people out of the map.
#
# Multi-query mode: comma-separated DYNLANG_QUERIES gets a separate
# composite per query. Single-query mode (DYNLANG_QUERY=foo) is also
# supported as a shortcut for backward compatibility.
DEFAULT_QUERIES = {
    "h1": "desk,keyboard,monitor,chair,floor",
    "h2": "table,balloon,chair,floor",
    "h3": "chair,table,floor,wall,plant,sofa",
    "h4": "ground,wall,sky,car,tree,building",
}
_query_env = os.environ.get("DYNLANG_QUERIES") or os.environ.get("DYNLANG_QUERY")
QUERIES = [q.strip() for q in
           (_query_env or DEFAULT_QUERIES.get(SCENE, "object")).split(",")
           if q.strip()]

# Language ON by default; set DYNLANG_LANG=0 to drop the 4th row.
USE_LANGUAGE = os.environ.get("DYNLANG_LANG", "1") == "1"

# Heatmap scoring mode:
#   "cosine" (default): raw cosine sim in [-1, 1] -> percentile-stretched.
#       Smoother/softer; less prone to amplifying noise into spurious hotspots
#       on small-training-set Gaussian language features.
#   "relevancy": LangSplat contrastive softmax vs canonical phrases (temp=50).
#       Sharper hotspots but very sensitive to noisy lang_feats.
HEATMAP_MODE = os.environ.get("DYNLANG_HEATMAP", "cosine").lower()

# Frames to capture. Default skews toward the early build-up phase --
# the most visible "progress" stage when the map is doubling in size
# every few keyframes. 5 frames keeps the figure narrow enough for a
# CVPR two-column page; override with DYNLANG_FRAMES if you want denser.
default_frames = "1,15,30,60,99"
CAPTURE_FRAMES = sorted({
    int(s) for s in os.environ.get("DYNLANG_FRAMES", default_frames).split(",") if s.strip()
})

# Canonical pose for the "progress" row. Defaults to the second-to-last
# capture frame (e.g. 60 with the default capture set) -- gives early
# snapshots a chance to have *something* in view, instead of all-black
# panels when the canonical pose is way ahead of where the camera was.
# Override with DYNLANG_CANONICAL=99 (or any int) to force a specific
# frame.
_canonical_env = os.environ.get("DYNLANG_CANONICAL")
if _canonical_env is not None and _canonical_env.strip():
    CANONICAL_FRAME = int(_canonical_env)
else:
    CANONICAL_FRAME = (CAPTURE_FRAMES[-2] if len(CAPTURE_FRAMES) >= 2
                       else CAPTURE_FRAMES[-1])

# Ablation hooks: env-var overrides for figure 9 (dynamic OFF) and
# figure 10 (broken language config = pre-fix). When set, also append a
# suffix to the output dir so we don't clobber the canonical run.
ABLATE_DYNAMIC_OFF = os.environ.get("DYNLANG_DYNAMIC_OFF", "0") in ("1", "true", "True")
ABLATE_BROKEN_AE   = os.environ.get("DYNLANG_BROKEN_AE",   "0") in ("1", "true", "True")
_OUT_SUFFIX = os.environ.get("DYNLANG_OUT_SUFFIX", "").strip()
if _OUT_SUFFIX:
    _OUT_NAME = f"filmstrip_{SCENE}_{_OUT_SUFFIX}"
else:
    _OUT_NAME = f"filmstrip_{SCENE}"
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures", _OUT_NAME)
os.makedirs(OUT_DIR, exist_ok=True)


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---- Scene loaders ------------------------------------------------------

def _build_cfg_bonn():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0
    cfg.dynamic.enabled = (not ABLATE_DYNAMIC_OFF)  # ablation: figure 9
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = False
    cfg.language.enabled = USE_LANGUAGE
    if USE_LANGUAGE:
        cfg.language.extract_every_n = 2
        cfg.language.autoencoder.warmup_frames = 30
        cfg.language.sam_checkpoint = os.path.join(
            PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"
        )
        # Ablation: figure 10 -- restore the pre-fix broken AE config.
        # latent_dim=16 + warmup_frames=100 made the AE fail to converge
        # on short BONN runs, leaving lang_feats near zero. We compare
        # the heatmap quality of this config against the fixed
        # latent_dim=128 + warmup=20 baseline.
        if ABLATE_BROKEN_AE:
            cfg.gaussians.lang_feat_dim = 16
            cfg.language.autoencoder.latent_dim = 16
            cfg.language.autoencoder.warmup_frames = 100
    cfg.slam.bootstrap.enabled = False
    return cfg


def _build_cfg_replica():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.scene = "room0"
    cfg.dataset.max_frames = N_REPLICA
    # Replica's native 680x1200 resolution OOMs on 8 GB GPUs during the
    # SAM2 mask step. Downscale lets the geometry-only run fit. Override
    # via DYNLANG_REPLICA_HW="<H>x<W>" if needed.
    hw_override = os.environ.get("DYNLANG_REPLICA_HW", "480x720")
    rh, rw = (int(s) for s in hw_override.lower().split("x"))
    cfg.dataset.image_height = rh
    cfg.dataset.image_width = rw
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = False
    cfg.language.enabled = USE_LANGUAGE
    if USE_LANGUAGE:
        cfg.language.extract_every_n = 5
        cfg.language.autoencoder.warmup_frames = 30
        cfg.language.sam_checkpoint = os.path.join(
            PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"
        )
    cfg.slam.bootstrap.enabled = False
    return cfg


def _build_cfg_h4():
    """In-the-wild Pixel 8 capture: portrait orientation, monocular
    depth from DepthAnything-V2, no ground-truth trajectory."""
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 854
    cfg.dataset.image_width = 480
    cfg.dataset.depth_scale = 5000.0
    cfg.dynamic.enabled = (not ABLATE_DYNAMIC_OFF)
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = False
    cfg.language.enabled = USE_LANGUAGE
    if USE_LANGUAGE:
        cfg.language.extract_every_n = 2
        cfg.language.autoencoder.warmup_frames = 30
        cfg.language.sam_checkpoint = os.path.join(
            PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"
        )
    cfg.slam.bootstrap.enabled = False
    return cfg


def _read_h4_intrinsics(intrinsics_path):
    """Parse a Pixel 8 intrinsics.txt file: 'fx fy cx cy' on the first
    non-comment line."""
    fx = fy = cx = cy = None
    with open(intrinsics_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 4:
                fx, fy, cx, cy = (float(x) for x in parts[:4])
                break
    if fx is None:
        raise ValueError(f"could not parse intrinsics from {intrinsics_path}")
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return {"K": K, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _load_scene():
    if SCENE == "h1":
        from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
        cfg = _build_cfg_bonn()
        ds = TUMDataset(
            data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_person_tracking"),
            height=480, width=640, depth_scale=5000.0,
            max_frames=N_BONN, stride=STRIDE_BONN,
        )
        return cfg, ds, get_bonn_intrinsics(), "BONN person_tracking"
    if SCENE == "h2":
        from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
        cfg = _build_cfg_bonn()
        ds = TUMDataset(
            data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_balloon"),
            height=480, width=640, depth_scale=5000.0,
            max_frames=N_BONN, stride=STRIDE_BONN,
        )
        return cfg, ds, get_bonn_intrinsics(), "BONN balloon"
    if SCENE == "h3":
        from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
        from pathlib import Path
        cfg = _build_cfg_replica()
        ds = ReplicaDataset(
            data_dir=str(Path(PROJECT_ROOT) / cfg.dataset.path / "room0"),
            height=cfg.dataset.image_height,
            width=cfg.dataset.image_width,
            depth_scale=cfg.dataset.depth_scale,
            max_frames=N_REPLICA,
        )
        K = get_replica_intrinsics(
            fx=cfg.camera.fx, fy=cfg.camera.fy,
            cx=cfg.camera.cx, cy=cfg.camera.cy,
            height=cfg.dataset.image_height, width=cfg.dataset.image_width,
        )
        return cfg, ds, K, "Replica room0"
    if SCENE == "h4":
        from dynlang_slam.data.tum import TUMDataset
        cfg = _build_cfg_h4()
        clip = os.environ.get("DYNLANG_H4_CLIP", "clip_a")
        h4_dir = os.path.join(PROJECT_ROOT, "data", "H4", clip)
        ds = TUMDataset(
            data_dir=h4_dir,
            height=cfg.dataset.image_height,
            width=cfg.dataset.image_width,
            depth_scale=cfg.dataset.depth_scale,
            max_frames=N_BONN, stride=STRIDE_BONN,
            has_gt=False,    # H4 has no ground-truth trajectory
        )
        K_dict = _read_h4_intrinsics(os.path.join(h4_dir, "intrinsics.txt"))
        K_dict["height"] = cfg.dataset.image_height
        K_dict["width"]  = cfg.dataset.image_width
        return cfg, ds, K_dict, f"H4 in-the-wild ({clip})"
    sys.exit(f"Unknown scene: {SCENE}")


# ---- Image helpers ------------------------------------------------------

def _to_hwc_u8(rgb_tensor) -> np.ndarray:
    """Coerce an arbitrary RGB tensor to (H, W, 3) uint8 numpy."""
    t = rgb_tensor.detach().cpu()
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)
    arr = t.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    return arr


def _overlay_mask(rgb_u8: np.ndarray, dynamic_mask_bool: np.ndarray) -> np.ndarray:
    """Blend semi-transparent red over pixels where mask is True (= dynamic)."""
    out = rgb_u8.astype(np.float32)
    red = np.array([220.0, 30.0, 30.0], dtype=np.float32)
    alpha = 0.45
    sel = dynamic_mask_bool[..., None]  # (H, W, 1)
    out = np.where(sel, (1 - alpha) * out + alpha * red, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_map(slam, gmap, frame_idx) -> np.ndarray:
    """Rasterize the current Gaussian map from the just-tracked pose."""
    pose_c2w = slam.estimated_poses[frame_idx].to(slam.device)
    return _render_map_from_pose(slam, gmap, pose_c2w)


def _render_map_from_pose(slam, gmap, pose_c2w) -> np.ndarray:
    """Rasterize the current Gaussian map from an arbitrary c2w pose."""
    viewmat = fast_se3_inverse(pose_c2w)
    with torch.no_grad():
        out = slam.renderer(
            gaussian_map=gmap,
            viewmat=viewmat,
            K=slam.K,
            width=slam.width,
            height=slam.height,
        )
    return _to_hwc_u8(out["rgb"])


def _render_belief_field(slam, gmap, frame_idx) -> np.ndarray:
    """Render the per-Gaussian dynamic_belief field as a (H, W) float
    image in [0, 1]. Used for figure 12 (belief evolution).

    Implementation: drop into gsplat directly with per-Gaussian colors
    set to the belief value broadcast to 3 channels. The rasterized
    output's red channel is then the alpha-weighted belief sum.

    Returns 0s if there are no Gaussians or no belief tensor.
    """
    if gmap.num_gaussians == 0 or not hasattr(gmap, "dynamic_belief") or \
            gmap.dynamic_belief is None or gmap.dynamic_belief.shape[0] == 0:
        return np.zeros((slam.height, slam.width), dtype=np.float32)

    from gsplat import rasterization
    pose_c2w = slam.estimated_poses[frame_idx].to(slam.device)
    viewmat = fast_se3_inverse(pose_c2w)

    params = gmap.get_activated_params()
    belief = gmap.dynamic_belief.detach().to(slam.device)  # (N,)
    # Broadcast belief to 3 channels so gsplat treats it as a "color"
    colors = belief.unsqueeze(-1).expand(-1, 3).contiguous()  # (N, 3)

    with torch.no_grad():
        rendered, _, _ = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"].squeeze(-1),
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=slam.K.unsqueeze(0),
            width=slam.width,
            height=slam.height,
            near_plane=0.01, far_plane=100.0,
            render_mode="RGB",
            packed=True,
            sh_degree=None,
            absgrad=False,
        )
    # Red channel = alpha-composited belief (channels are identical)
    belief_map = rendered[0, :, :, 0].clamp(0, 1).cpu().numpy()
    return belief_map.astype(np.float32)


def _snapshot_map(gmap) -> dict:
    """Deep-copy the renderable state of a GaussianMap. Returns a dict
    of cloned tensors -- enough for later replay through the renderer.

    We snapshot raw (un-activated) parameters; activations (exp on scale,
    sigmoid on opacity, normalize on quat) happen inside the renderer
    via get_activated_params(), so storing raw is sufficient and safe.
    """
    snap = {
        "n": gmap.num_gaussians,
        "means": gmap.means.data.detach().clone(),
        "scales": gmap.scales.data.detach().clone(),
        "quats": gmap.quats.data.detach().clone(),
        "opacities": gmap.opacities.data.detach().clone(),
        "colors": gmap.colors.data.detach().clone(),
    }
    if hasattr(gmap, "lang_feats") and gmap.lang_feats is not None \
            and gmap.lang_feats.shape[0] > 0:
        snap["lang_feats"] = gmap.lang_feats.data.detach().clone()
    else:
        snap["lang_feats"] = None
    # Per-Gaussian dynamic-belief field. Snapshotted so figure 12 can
    # show how belief grows on contaminated Gaussians over time.
    if hasattr(gmap, "dynamic_belief") and gmap.dynamic_belief is not None \
            and gmap.dynamic_belief.shape[0] > 0:
        snap["dynamic_belief"] = gmap.dynamic_belief.detach().clone()
    else:
        snap["dynamic_belief"] = None
    return snap


def _restore_map(gmap, snap) -> None:
    """Replace gmap state with a snapshot's tensors. Resizes the
    contamination_count / dynamic_belief buffers so the renderer's
    internal sanity-checks (which rely on consistent N) still pass.

    Note: this DOES discard any optimizer state that was attached to
    these Parameters. That's fine -- we only render, we don't optimize."""
    import torch.nn as nn
    gmap.means = nn.Parameter(snap["means"].clone())
    gmap.scales = nn.Parameter(snap["scales"].clone())
    gmap.quats = nn.Parameter(snap["quats"].clone())
    gmap.opacities = nn.Parameter(snap["opacities"].clone())
    gmap.colors = nn.Parameter(snap["colors"].clone())
    if snap["lang_feats"] is not None:
        gmap.lang_feats = nn.Parameter(snap["lang_feats"].clone())
    N = snap["n"]
    gmap._num_gaussians = N
    if hasattr(gmap, "contamination_count"):
        gmap.contamination_count = torch.zeros(
            N, dtype=torch.long, device=gmap.device
        )
    if hasattr(gmap, "dynamic_belief"):
        # If the snapshot includes a saved belief tensor, use it (figure
        # 12). Otherwise zero. Critical that this restores AFTER setting
        # _num_gaussians so the buffer's first dim matches.
        if snap.get("dynamic_belief") is not None and \
                snap["dynamic_belief"].shape[0] == N:
            gmap.dynamic_belief = snap["dynamic_belief"].clone().to(gmap.device)
        else:
            gmap.dynamic_belief = torch.zeros(
                N, dtype=torch.float32, device=gmap.device
            )


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR between two uint8 RGB images (H, W, 3). Higher = closer."""
    a32 = a.astype(np.float64)
    b32 = b.astype(np.float64)
    mse = np.mean((a32 - b32) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


# Prompt-ensemble templates -- LERF / CLIP zero-shot best-practice. Encoding
# all of these in CLIP space and averaging makes a single-word query like
# "monitor" much more robust than the bare token, because CLIP was trained
# on caption-like text, not bare nouns. We average in CLIP space (768-d) and
# re-normalize *before* the autoencoder, so the AE sees a denoised query.
_PROMPT_TEMPLATES = [
    "{}",
    "a {}",
    "a photo of a {}",
    "a photo of the {}",
    "an image of a {}",
    "the {} in the room",
    "a close-up of a {}",
]

# Richer canonical set. LERF's original 4 ("object", "things", "stuff",
# "texture") collapse easily on small training sets. Adding scene-agnostic
# distractors gives the contrastive softmax more headroom.
_CANONICAL_PHRASES = [
    "object", "things", "stuff", "texture",
    "background", "wall", "scene", "surface",
]


def _encode_query(slam, query_text: str) -> torch.Tensor | None:
    """Encode the text query through CLIP (with prompt ensembling) -> AE.

    Multi-synonym queries: pass a `|`-separated string like
    "monitor|screen|display" to take the *max* relevancy over the synonym
    bag (Bag-of-Embeddings lite). This is way more robust than any single
    word for objects CLIP wasn't supervised on with that exact label.
    """
    if not getattr(slam, "_lang_initialized", False) or slam._autoencoder is None:
        return None
    slam._clip_extractor.model.to(slam.device)
    slam._clip_extractor.device = slam.device

    # Split synonyms; build prompt-ensemble per synonym, then stack
    synonyms = [s.strip() for s in query_text.split("|") if s.strip()]
    if not synonyms:
        synonyms = [query_text]

    syn_latents = []
    for s in synonyms:
        prompts = [tpl.format(s) for tpl in _PROMPT_TEMPLATES]
        clip_feats = slam._clip_extractor.encode_texts(prompts)  # (T, 768)
        # Mean in CLIP space, then renormalize
        mean_clip = torch.nn.functional.normalize(
            clip_feats.mean(dim=0, keepdim=True), dim=-1
        )  # (1, 768)
        latent = slam._autoencoder.encode(mean_clip.to(slam.device)).squeeze(0)  # (D,)
        syn_latents.append(latent)
    text_latent = torch.stack(syn_latents, dim=0)  # (S, D), S=#synonyms

    # Canonical phrases: also use prompt-ensemble (averaged in CLIP space)
    canon_latents = []
    for c in _CANONICAL_PHRASES:
        prompts = [tpl.format(c) for tpl in _PROMPT_TEMPLATES]
        clip_feats = slam._clip_extractor.encode_texts(prompts)
        mean_clip = torch.nn.functional.normalize(
            clip_feats.mean(dim=0, keepdim=True), dim=-1
        )
        canon_latents.append(
            slam._autoencoder.encode(mean_clip.to(slam.device)).squeeze(0)
        )
    canon_latents = torch.stack(canon_latents, dim=0)  # (C, D)

    # Offload CLIP back to CPU
    slam._clip_extractor.model.to("cpu")
    slam._clip_extractor.device = "cpu"
    torch.cuda.empty_cache()
    return text_latent, canon_latents


def _render_query_heatmap(slam, gmap, frame_idx, query_latent, canon_latents) -> np.ndarray:
    """Rasterize lang_feats from this frame's pose, then per-pixel
    similarity vs the query text. Returns (H, W) float.

    Two scoring modes (selected by HEATMAP_MODE):
        "cosine"    -> raw cosine in [-1, 1]. Soft, faithful to feature
                       geometry, less prone to noise amplification.
        "relevancy" -> LangSplat-style temperature-scaled contrastive
                       softmax in [0, 1] (matches SLAMPipeline.query_3d).
    """
    if query_latent is None:
        return np.zeros((slam.height, slam.width), dtype=np.float32)
    pose_c2w = slam.estimated_poses[frame_idx].to(slam.device)
    viewmat = fast_se3_inverse(pose_c2w)
    with torch.no_grad():
        out = slam.renderer(
            gaussian_map=gmap,
            viewmat=viewmat,
            K=slam.K,
            width=slam.width,
            height=slam.height,
            render_lang=True,
        )
        feats = out.get("lang_feats")  # (H, W, D)
        if feats is None:
            return np.zeros((slam.height, slam.width), dtype=np.float32)
        H, W, D = feats.shape
        feats_flat = feats.view(-1, D)
        feats_norm = torch.nn.functional.normalize(feats_flat, dim=-1)

        # query_latent is now (S, D): synonyms-as-bag.
        # Per-pixel similarity = max over synonyms (Bag-of-Embeddings lite).
        if query_latent.ndim == 1:
            query_latent = query_latent.unsqueeze(0)
        q_norm = torch.nn.functional.normalize(query_latent, dim=-1)  # (S, D)
        sim_q_all = feats_norm @ q_norm.T  # (HW, S)
        sim_q = sim_q_all.max(dim=-1).values  # (HW,)

        if HEATMAP_MODE == "relevancy":
            # LERF-style: contrast against the *most-confusable* canonical
            # (max over canonicals, then softmax with high temperature).
            canon_norm = torch.nn.functional.normalize(canon_latents, dim=-1)
            sim_c = (feats_norm @ canon_norm.T).max(dim=-1).values
            temperature = 50.0
            score = torch.exp(temperature * sim_q) / (
                torch.exp(temperature * sim_q) + torch.exp(temperature * sim_c)
            )
        else:  # "cosine"
            score = sim_q

        score = score.view(H, W).cpu().numpy()
    return score.astype(np.float32)


def _overlay_heatmap(rgb_u8: np.ndarray, heatmap: np.ndarray,
                     vmin: float | None = None, vmax: float | None = None,
                     alpha: float = 0.55) -> np.ndarray:
    """Blend a 'turbo' colormap of heatmap over the input. Robust
    percentile-based contrast stretch so a low-magnitude query still
    shows structure."""
    h = heatmap.astype(np.float32)
    if vmin is None:
        vmin = float(np.percentile(h, 60))
    if vmax is None:
        vmax = float(np.percentile(h, 99))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    h_n = np.clip((h - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    colored = (cmap(h_n)[..., :3] * 255).astype(np.float32)  # (H, W, 3)
    # Mask blend: only paint pixels above a low threshold so cold areas
    # show the underlying RGB. Smooth transition via the normalized score.
    blend_alpha = (h_n[..., None] * alpha)
    out = (1 - blend_alpha) * rgb_u8.astype(np.float32) + blend_alpha * colored
    return np.clip(out, 0, 255).astype(np.uint8)


def _detect_dynamic(slam, rgb_tensor) -> np.ndarray:
    """Re-run YOLO + merge to bool dynamic mask. (No temporal filter --
    raw detection is what the prof's audience will recognize.)"""
    if not getattr(slam, "dynamic_enabled", False):
        return np.zeros((slam.height, slam.width), dtype=bool)
    if getattr(slam, "_dynamic_detector", None) is None:
        # Force lazy-init by calling the same path the pipeline uses
        slam._init_dynamic_pipeline()
    detector = slam._dynamic_detector
    mask = detector.detect_and_merge(rgb_tensor)  # (H, W) bool tensor
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return mask.astype(bool)


# ---- Main ---------------------------------------------------------------

def main():
    _seed_everything(SEED)
    cfg, dataset, intrinsics, display_name = _load_scene()

    print(f"\n{'='*72}")
    print(f" filmstrip  scene={SCENE} ({display_name})  seed={SEED}  "
          f"capture_frames={CAPTURE_FRAMES}")
    print(f"          language={'ON' if USE_LANGUAGE else 'OFF'}"
          f"  heatmap_mode={HEATMAP_MODE}"
          f"{f'  queries=' + repr(QUERIES) if USE_LANGUAGE else ''}")
    print(f"{'='*72}")

    device = "cuda"
    gmap = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
    slam.process_first_frame(gmap, dataset[0])

    captured = {}  # frame_idx -> dict{rgb, render, mask_overlay,
                   #                   snapshot, render_canonical, psnr_*,
                   #                   query_overlays: {q: arr},
                   #                   query_heats: {q: arr}}

    def _capture(idx, rgb_tensor):
        rgb_u8 = _to_hwc_u8(rgb_tensor)
        render = _render_map(slam, gmap, idx)
        mask = _detect_dynamic(slam, rgb_tensor)
        # Snapshot map state -- needed for canonical-view "progress" row.
        snap = _snapshot_map(gmap)
        # Render the dynamic-belief field at this frame (figure 12).
        # Use percentile-stretched overlay (the default in _overlay_heatmap)
        # so sparse high-belief regions visibly pop -- absolute belief
        # values cluster low even on contaminated Gaussians, since the
        # alpha-composited sum splits across all Gaussians on the ray.
        belief = _render_belief_field(slam, gmap, idx)
        if belief.max() > 1e-6:
            belief_overlay = _overlay_heatmap(rgb_u8, belief, alpha=0.75)
        else:
            # All-zero belief (e.g. frame 1 before any contamination):
            # show the input RGB without any tint.
            belief_overlay = rgb_u8.copy()
        panels = dict(
            rgb=rgb_u8,
            render=render,
            mask_overlay=_overlay_mask(rgb_u8, mask),
            belief=belief,
            belief_overlay=belief_overlay,
            n_gauss=gmap.num_gaussians,
            snapshot=snap,
            render_canonical=None,         # filled in post-SLAM
            psnr_render=_psnr(rgb_u8, render),  # vs this frame's input
            psnr_canonical=None,           # vs canonical-frame input (post-SLAM)
            query_overlays={},
            query_heats={},
        )
        captured[idx] = panels
        b_max = float(belief.max()) if belief.size else 0.0
        print(f"  [frame {idx}] captured  ({gmap.num_gaussians} gaussians, "
              f"PSNR={panels['psnr_render']:.2f} dB, belief_max={b_max:.3f})")

    if 0 in CAPTURE_FRAMES:
        _capture(0, dataset[0]["rgb"])

    for i in range(1, len(dataset)):
        slam.process_frame(gmap, dataset[i], use_gt_pose=False)
        if i in CAPTURE_FRAMES:
            _capture(i, dataset[i]["rgb"])

    # ---- Canonical-view "progress" row -------------------------------
    # Render every snapshot from a single fixed pose. Same view,
    # growing map => map fill-in is unambiguous. Uses GT input at the
    # canonical frame as PSNR reference.
    #
    # The canonical pose defaults to the SECOND-to-last capture frame
    # rather than the very last, so early snapshots aren't rendered
    # against a viewpoint the camera hadn't reached yet (which yields
    # all-black panels and looks like a bug at first glance).
    if CANONICAL_FRAME in captured:
        canonical_idx = CANONICAL_FRAME
    else:
        # Fall back to whichever captured frame is closest to the requested.
        canonical_idx = min(captured.keys(),
                            key=lambda f: abs(f - CANONICAL_FRAME))
        print(f"  [progress] canonical frame {CANONICAL_FRAME} not in "
              f"captured set; using closest frame {canonical_idx}")
    canonical_pose = slam.estimated_poses[canonical_idx].to(slam.device).clone()
    canonical_rgb_u8 = captured[canonical_idx]["rgb"]
    final_snap = _snapshot_map(gmap)  # save final state so we can restore
    print(f"  [progress] rendering canonical view (frame={canonical_idx} pose) "
          f"for each snapshot...")
    try:
        for fid, panels in captured.items():
            _restore_map(gmap, panels["snapshot"])
            cano_render = _render_map_from_pose(slam, gmap, canonical_pose)
            panels["render_canonical"] = cano_render
            panels["psnr_canonical"] = _psnr(canonical_rgb_u8, cano_render)
            print(f"    frame {fid}: {panels['n_gauss']} gauss, "
                  f"canonical PSNR={panels['psnr_canonical']:.2f} dB")
    finally:
        # Always restore the FINAL map state so language heatmaps below
        # query the trained map, not whichever snapshot was last loaded.
        _restore_map(gmap, final_snap)

    # ---- Save final Gaussian-map checkpoint + trajectory for the
    # ---- tracking-and-mapping fly-by visualisation. -------------------
    ckpt = {
        "snapshot": {
            k: (v.detach().cpu() if hasattr(v, "detach") else v)
            for k, v in final_snap.items()
        },
        "estimated_poses": torch.stack([
            p.detach().cpu() for p in slam.estimated_poses
        ]),
        "K": slam.K.detach().cpu(),
        "width": int(slam.width),
        "height": int(slam.height),
        "scene": SCENE,
    }
    ckpt_path = os.path.join(OUT_DIR, f"map_ckpt_{SCENE}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"  [ckpt] saved Gaussian-map checkpoint to {ckpt_path} "
          f"({ckpt['estimated_poses'].shape[0]} poses, "
          f"{ckpt['snapshot']['n']} gaussians)")

    # ---- Language heatmaps (one per query, all from same SLAM run) ----
    rendered_queries = []  # queries that successfully produced heatmaps
    if USE_LANGUAGE:
        if not getattr(slam, "_lang_initialized", False) or slam._autoencoder is None:
            print("  [lang] WARNING: language pipeline did not initialize "
                  "(too few frames? autoencoder warmup not reached?). "
                  "Skipping query heatmap rows.")
        else:
            for q in QUERIES:
                encoded = _encode_query(slam, q)
                if encoded is None:
                    continue
                query_latent, canon_latents = encoded
                print(f"  [lang] query={q!r}  rendering heatmaps...")
                for fid, panels in captured.items():
                    heat = _render_query_heatmap(
                        slam, gmap, fid, query_latent, canon_latents
                    )
                    panels["query_heats"][q] = heat
                    panels["query_overlays"][q] = _overlay_heatmap(
                        panels["rgb"], heat
                    )
                    print(f"    frame {fid}: heat range "
                          f"[{heat.min():.3f}, {heat.max():.3f}] "
                          f"mean={heat.mean():.3f}")
                rendered_queries.append(q)

    # ---- Save per-frame PNGs ------------------------------------------
    for fid, panels in captured.items():
        Image.fromarray(panels["rgb"]).save(
            os.path.join(OUT_DIR, f"frame_{fid:03d}_rgb.png"))
        Image.fromarray(panels["render"]).save(
            os.path.join(OUT_DIR, f"frame_{fid:03d}_render.png"))
        Image.fromarray(panels["mask_overlay"]).save(
            os.path.join(OUT_DIR, f"frame_{fid:03d}_mask.png"))
        if panels.get("belief_overlay") is not None:
            Image.fromarray(panels["belief_overlay"]).save(
                os.path.join(OUT_DIR, f"frame_{fid:03d}_belief.png"))
        if panels.get("render_canonical") is not None:
            Image.fromarray(panels["render_canonical"]).save(
                os.path.join(OUT_DIR, f"frame_{fid:03d}_canonical.png"))
        for q, arr in panels["query_overlays"].items():
            qslug = q.replace(" ", "_").replace("|", "-")
            Image.fromarray(arr).save(
                os.path.join(OUT_DIR, f"frame_{fid:03d}_lang_{qslug}.png"))

    # ---- Composite filmstrips -----------------------------------------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    frame_ids = sorted(captured.keys())
    cols = len(frame_ids)

    def _save_composite(out_basename, row_labels, panel_arrays_per_row,
                        suptitle=None, col_titles=None):
        """panel_arrays_per_row: list of lists, each inner list = images per col.
        col_titles: optional list[str] aligned with frame_ids; defaults to
                    "frame N\n(M.Mk Gauss, PSNR=X.XdB)" if available."""
        rows = len(row_labels)
        fig, axs = plt.subplots(
            rows, cols,
            figsize=(1.6 * cols + 0.7, 1.6 * rows + 0.3),
            constrained_layout=True,
        )
        if rows == 1:
            axs = np.array([axs])
        if cols == 1:
            axs = axs.reshape(rows, 1)
        if col_titles is None:
            col_titles = []
            for fid in frame_ids:
                ng = captured[fid]["n_gauss"]
                psnr = captured[fid].get("psnr_render")
                if psnr is not None:
                    col_titles.append(
                        f"frame {fid}\n{ng/1000:.1f}k Gauss, {psnr:.1f} dB"
                    )
                else:
                    col_titles.append(f"frame {fid}\n({ng/1000:.1f}k Gauss)")
        for r in range(rows):
            for c, fid in enumerate(frame_ids):
                ax = axs[r, c]
                ax.imshow(panel_arrays_per_row[r][c])
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
                if r == 0:
                    ax.set_title(col_titles[c], fontsize=8)
                if c == 0:
                    ax.set_ylabel(row_labels[r], fontsize=9, fontweight="bold")
        fig.suptitle(
            suptitle or
            f"DynLang-SLAM on {display_name}: tracking + mapping over time",
            fontsize=10, y=1.02,
        )
        for ext in ("pdf", "png"):
            out = os.path.join(OUT_DIR, f"{out_basename}.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  wrote {out}")
        plt.close(fig)

    base_rows = ["Input", "Render", "Dynamic mask"]
    base_arrays = [
        [captured[f]["rgb"] for f in frame_ids],
        [captured[f]["render"] for f in frame_ids],
        [captured[f]["mask_overlay"] for f in frame_ids],
    ]

    # Summary stats for the suptitle: map growth + PSNR improvement.
    n_first = captured[frame_ids[0]]["n_gauss"]
    n_last = captured[frame_ids[-1]]["n_gauss"]
    psnr_first = captured[frame_ids[0]]["psnr_render"]
    psnr_last = captured[frame_ids[-1]]["psnr_render"]
    progress_suptitle = (
        f"DynLang-SLAM on {display_name}: map "
        f"{n_first/1000:.1f}k → {n_last/1000:.1f}k Gaussians, "
        f"render PSNR {psnr_first:.1f} → {psnr_last:.1f} dB"
    )

    # 1) Core composite (no language).
    _save_composite(
        f"filmstrip_{SCENE}", base_rows, base_arrays,
        suptitle=progress_suptitle,
    )

    # 1b) "Progress" composite: input + render-from-this-pose +
    #     render-from-canonical-pose. Same canonical view across all
    #     timesteps shows the map fill-in unambiguously.
    if all(captured[f]["render_canonical"] is not None for f in frame_ids):
        # Custom column titles: top row uses per-frame PSNR, but call out
        # canonical PSNR explicitly so the progression is visible.
        prog_titles = []
        for fid in frame_ids:
            ng = captured[fid]["n_gauss"]
            cp = captured[fid]["psnr_canonical"]
            prog_titles.append(
                f"frame {fid}\n{ng/1000:.1f}k Gauss"
                + (f"\ncanon. PSNR {cp:.1f} dB" if cp is not None else "")
            )
        cano_first = captured[frame_ids[0]]["psnr_canonical"]
        cano_last = captured[frame_ids[-1]]["psnr_canonical"]
        prog_supt = (
            f"DynLang-SLAM map build-up on {display_name}: "
            f"{n_first/1000:.1f}k → {n_last/1000:.1f}k Gaussians, "
            f"canonical-view PSNR {cano_first:.1f} → {cano_last:.1f} dB"
        )
        _save_composite(
            f"filmstrip_{SCENE}_progress",
            ["Input", "Render @ this pose", f"Render @ frame-{canonical_idx} pose"],
            [
                [captured[f]["rgb"] for f in frame_ids],
                [captured[f]["render"] for f in frame_ids],
                [captured[f]["render_canonical"] for f in frame_ids],
            ],
            suptitle=prog_supt,
            col_titles=prog_titles,
        )

    # 2) Per-query 4-row composites (paper-friendly singles).
    for q in rendered_queries:
        qslug = q.replace(" ", "_").replace("|", "-")
        _save_composite(
            f"filmstrip_{SCENE}_lang_{qslug}",
            base_rows + [f"Query: '{q}'"],
            base_arrays + [[captured[f]["query_overlays"][q] for f in frame_ids]],
        )

    # 3) Master multi-query composite. Includes canonical-view row
    #    (when available) so the master also tells the progress story.
    have_canonical = all(
        captured[f]["render_canonical"] is not None for f in frame_ids
    )
    master_rows = list(base_rows)
    master_arrays = list(base_arrays)
    if have_canonical:
        master_rows.append(f"Render @ frame-{canonical_idx} pose")
        master_arrays.append(
            [captured[f]["render_canonical"] for f in frame_ids]
        )
    if rendered_queries:
        master_rows += [f"Query: '{q}'" for q in rendered_queries]
        master_arrays += [
            [captured[f]["query_overlays"][q] for f in frame_ids]
            for q in rendered_queries
        ]
    if len(master_rows) > len(base_rows):  # at least one extra signal
        _save_composite(
            f"filmstrip_{SCENE}_master", master_rows, master_arrays,
            suptitle=progress_suptitle,
        )

    print(f"\n[done] {len(rendered_queries)} queries rendered + per-frame PNGs "
          f"+ composites saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
