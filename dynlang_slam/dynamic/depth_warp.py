"""Class-agnostic dynamic mask via depth consistency across frames.

Ports the depth-warp mask from DG-SLAM (Xu et al., NeurIPS 2024).

Mechanism
---------
For each recent frame `i` with depth map D_i and pose T_i, we back-project
every pixel of the current frame `j` (depth D_j, pose T_j) into 3D, transform
it into i's camera frame, project to i's image, and compare the sampled
depth D_i(p_i) to the transformed Z coordinate. A mismatch above `threshold`
is evidence that the scene content at that pixel moved between i and j —
class-agnostic motion detection.

Because single-frame warp masks are noisy (occlusions, depth holes), we
require a pixel to be flagged dynamic **in all N** recent frames before
trusting it — this is the intersection step in DG-SLAM Eq. 7.

Reference
---------
DG-SLAM Eq. 5-7:
  p_{i→j}          = K · T_ji · ( K^{-1} · D_i(p_i) · p_i^homo )
  dyn_{j,i}        = | D_j(p_{i→j}) − D_i(p_i) | > e_th
  M_j^dynamic      = ∩_{i in window} dyn_{j,i}     # intersect → confident
  M_j^final        = M_j^dynamic  ∪  M_j^semantic  # union with YOLO

We implement the equivalent formulation but index by the CURRENT frame's
pixel grid (simpler to consume downstream): reproject j's pixels INTO i,
sample i's depth at the reprojected location, and compare against j's
depth transformed into i's frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class _WarpFrame:
    """A single frame retained in the warp buffer."""
    depth: torch.Tensor    # (H, W) meters, 0 where invalid
    pose: torch.Tensor     # (4, 4) camera-to-world, SE(3)


class DepthWarpBuffer:
    """Fixed-size FIFO of recent (depth, pose) pairs for warp-mask computation."""

    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self._frames: deque[_WarpFrame] = deque(maxlen=window_size)

    def push(self, depth: torch.Tensor, pose: torch.Tensor) -> None:
        """Store a frame. Depth should be (H, W) float in meters, pose (4, 4) c2w."""
        if depth.dim() == 3:
            depth = depth.squeeze(0) if depth.shape[0] == 1 else depth.squeeze(-1)
        self._frames.append(_WarpFrame(depth=depth.detach(), pose=pose.detach()))

    def __len__(self) -> int:
        return len(self._frames)

    def frames(self) -> list[_WarpFrame]:
        return list(self._frames)


@torch.no_grad()
def _pairwise_dynamic_mask(
    D_j: torch.Tensor,         # (H, W) current frame depth
    T_j: torch.Tensor,         # (4, 4) current pose, c2w
    D_i: torch.Tensor,         # (H, W) reference frame depth
    T_i: torch.Tensor,         # (4, 4) reference pose, c2w
    K: torch.Tensor,           # (3, 3) intrinsics
    threshold: float = 0.6,    # DG-SLAM default
) -> torch.Tensor:
    """Return a (H, W) bool mask: True ⇒ pixel is dynamic under this pair.

    Pixels with invalid depth (0) on either side, or whose reprojection falls
    outside i's image, are returned as False (not dynamic) — we don't have
    evidence to flag them.
    """
    H, W = D_j.shape
    device = D_j.device

    # Build pixel grid for frame j: (H, W, 3) homogeneous
    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(us)
    p_j = torch.stack([us, vs, ones], dim=-1)          # (H, W, 3)

    # Back-project j's pixels to 3D in j's camera coords
    K_inv = torch.inverse(K)
    rays = p_j @ K_inv.T                                # (H, W, 3), unit-z rays
    P_j_cam = rays * D_j.unsqueeze(-1)                  # (H, W, 3), scaled by depth

    # Transform 3D points from j's camera -> i's camera via c2w poses
    # P_world = T_j · P_j_cam    ⇒  P_i_cam = T_i^{-1} · T_j · P_j_cam
    T_ij = torch.inverse(T_i) @ T_j                     # (4, 4)
    R_ij = T_ij[:3, :3]
    t_ij = T_ij[:3, 3]
    P_i_cam = P_j_cam @ R_ij.T + t_ij                   # (H, W, 3)

    # Predicted depth in i's frame is simply P_i_cam.z
    z_i_pred = P_i_cam[..., 2]                          # (H, W)

    # Project into i's image
    u_i = (P_i_cam[..., 0] * K[0, 0] / z_i_pred.clamp(min=1e-6)) + K[0, 2]
    v_i = (P_i_cam[..., 1] * K[1, 1] / z_i_pred.clamp(min=1e-6)) + K[1, 2]

    # Sample D_i at (u_i, v_i) with bilinear interp using grid_sample.
    # grid_sample wants normalized coords in [-1, 1].
    norm_u = 2.0 * u_i / max(W - 1, 1) - 1.0
    norm_v = 2.0 * v_i / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(0)   # (1, H, W, 2)
    D_i_sampled = F.grid_sample(
        D_i.unsqueeze(0).unsqueeze(0),                   # (1, 1, H, W)
        grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    ).squeeze(0).squeeze(0)                              # (H, W)

    # Validity: depth must be positive on both sides, reprojection in-bounds,
    # and z_i_pred must be in front of i's camera.
    in_bounds = (norm_u.abs() < 1.0) & (norm_v.abs() < 1.0)
    valid = in_bounds & (D_j > 0) & (D_i_sampled > 0) & (z_i_pred > 0)

    # Residual: absolute depth disagreement
    residual = (D_i_sampled - z_i_pred).abs()

    return valid & (residual > threshold)


@torch.no_grad()
def fused_motion_mask(
    current_depth: torch.Tensor,
    current_pose: torch.Tensor,
    buffer: DepthWarpBuffer,
    K: torch.Tensor,
    semantic_mask: torch.Tensor | None = None,
    threshold: float = 0.6,
    min_window: int = 2,
) -> torch.Tensor | None:
    """Build the DG-SLAM fused motion mask for the current frame.

    Args:
        current_depth: (H, W) depth map for the current frame.
        current_pose: (4, 4) camera-to-world pose estimate for current frame
            (use the velocity-init estimate; we don't need the tracked pose).
        buffer: `DepthWarpBuffer` of recent frames.
        K: (3, 3) intrinsics.
        semantic_mask: optional (H, W) bool/float, True/1 where YOLO says
            dynamic. The warp result is UNIONed with this.
        threshold: absolute depth disagreement (meters) to flag a pixel.
        min_window: need at least this many buffered frames before we return
            a warp-based mask. If fewer, fall back to `semantic_mask` only
            (returning None if that's also absent).

    Returns:
        (H, W) float tensor in {0.0, 1.0} where 1 = STATIC (pass-through
        for tracker), 0 = DYNAMIC (exclude from loss). Matches the convention
        already used by the rest of the pipeline: `mask` kwarg on
        `Tracker.track()` is 1=keep, 0=ignore.

        Returns `None` if neither source is available.
    """
    frames = buffer.frames()
    if current_depth.dim() == 3:
        current_depth = (
            current_depth.squeeze(0)
            if current_depth.shape[0] == 1 else current_depth.squeeze(-1)
        )

    H, W = current_depth.shape
    device = current_depth.device

    warp_dynamic = None
    if len(frames) >= min_window:
        # Intersection of per-pair dynamic masks across the window.
        # Start with all-True and AND in each pair's mask.
        warp_dynamic = torch.ones(H, W, dtype=torch.bool, device=device)
        for f in frames:
            pair_dyn = _pairwise_dynamic_mask(
                current_depth, current_pose, f.depth, f.pose, K, threshold,
            )
            warp_dynamic = warp_dynamic & pair_dyn

    # Bring semantic mask to the expected convention: 1 = dynamic (so we can
    # OR it with warp_dynamic). Incoming semantic_mask uses 1 = static,
    # 0 = dynamic (that's how Tracker.track receives it today) — invert.
    semantic_dynamic = None
    if semantic_mask is not None:
        sm = semantic_mask
        if sm.dtype != torch.bool:
            sm = sm > 0.5
        # sm True means STATIC (tracker input convention). Negate to dynamic.
        semantic_dynamic = ~sm

    if warp_dynamic is None and semantic_dynamic is None:
        return None
    if warp_dynamic is None:
        dynamic = semantic_dynamic
    elif semantic_dynamic is None:
        dynamic = warp_dynamic
    else:
        dynamic = warp_dynamic | semantic_dynamic

    # Convert dynamic -> static-float mask (1 = keep, 0 = exclude).
    static_mask = (~dynamic).float()
    return static_mask
