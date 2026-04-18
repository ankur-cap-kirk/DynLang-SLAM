"""Class-agnostic dynamic mask via point tracking (CoTracker2 / PIPs lineage).

Covered in EEE 515 Lecture 27 (Warping and Tracking) as the modern
occlusion-robust alternative to optical flow and classical features.

Mechanism
---------
We maintain a sliding buffer of the last T frames (RGB + depth + tracked
pose). When full, we:

1. Seed a 2D grid of query points on the OLDEST frame in the buffer (frame 0
   of the window). This frame has a known tracked pose and depth.
2. Track those points forward across all T frames with CoTracker2, getting
   observed 2D trajectories + per-frame visibility.
3. For each query point, compute the PREDICTED-STATIC 2D position on each
   frame: back-project through K and depth_0 at the seed, transform via
   T_i^{-1} @ T_0, reproject through K.
4. Compute the per-frame residual in pixels: || observed - predicted ||.
5. Median-correct across the whole grid to remove camera-egomotion bias
   (a DC offset in residuals — residual shared by ALL points is attributable
   to pose noise, not motion).
6. Aggregate over the window via max-over-frames.
7. Classify each query point as dynamic iff max-residual > thresh_px.
8. Upsample sparse 20x20 dynamic classification to dense (H, W) mask via
   nearest-neighbor, then dilate with a small kernel so the mask covers
   neighborhoods around the classified points.

Why this is expected to work where intervention B (depth-warp) failed
--------------------------------------------------------------------
- CoTracker's OBSERVATIONS are pose-independent (pure RGB).
- Residual is in pixels, not depth. Pose error of 5 cm @ 3 m depth gives
  roughly (fx * 0.05 / 3) = 10 px predicted-position bias — but bias is
  shared across the grid and cancels under median-correction.
- We use TRACKED poses for reprojection, not velocity-init.
- No intersection-AND step that erases true positives.

References
----------
- Harley, Fang, Fragkiadaki. Particle Video Revisited: Tracking Through
  Occlusions Using Point Trajectories. ECCV 2022.
- Karaev, Rocco, Graham, Laptev, Feichtenhofer. CoTracker: It is Better
  to Track Together. 2023.
- EEE 515 Lecture 27 (Warping and Tracking).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class _TrackFrame:
    rgb: torch.Tensor     # (3, H, W) uint8 or float in [0,255], on GPU
    depth: torch.Tensor   # (H, W) meters
    pose: torch.Tensor    # (4, 4) camera-to-world (SE(3))


class PointTrackBuffer:
    """Fixed-size FIFO of recent (rgb, depth, tracked-pose) frames."""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self._frames: deque[_TrackFrame] = deque(maxlen=window_size)

    def push(self, rgb: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor) -> None:
        if rgb.dim() == 4:
            rgb = rgb.squeeze(0)
        if rgb.shape[0] != 3 and rgb.shape[-1] == 3:
            rgb = rgb.permute(2, 0, 1)
        rgb = rgb.detach().float()
        if rgb.max() <= 1.01:
            rgb = rgb * 255.0
        if depth.dim() == 3:
            depth = depth.squeeze(0) if depth.shape[0] == 1 else depth.squeeze(-1)
        self._frames.append(_TrackFrame(
            rgb=rgb.detach(),
            depth=depth.detach(),
            pose=pose.detach(),
        ))

    def __len__(self) -> int:
        return len(self._frames)

    def frames(self) -> list[_TrackFrame]:
        return list(self._frames)

    def is_full(self) -> bool:
        return len(self._frames) >= self.window_size


def _make_query_grid(H: int, W: int, grid: int, margin: int = 10) -> torch.Tensor:
    """Return (N, 2) tensor of (x, y) query points as a regular grid."""
    ys = torch.linspace(margin, H - margin, grid)
    xs = torch.linspace(margin, W - margin, grid)
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([uu.flatten(), vv.flatten()], dim=-1)  # (N, 2)


def _predict_static_positions(
    query_xy: torch.Tensor,   # (N, 2) query (u, v) on frame 0
    depth0:   torch.Tensor,   # (H, W) depth of frame 0
    poses:    list[torch.Tensor],  # T poses, c2w, frame 0 is poses[0]
    K:        torch.Tensor,   # (3, 3)
) -> torch.Tensor:
    """Return (T, N, 2) predicted (u, v) positions if each point were static."""
    device = K.device
    N = query_xy.shape[0]
    T = len(poses)

    # Sample depth at query points (nearest)
    us = query_xy[:, 0].clamp(0, depth0.shape[1] - 1).long()
    vs = query_xy[:, 1].clamp(0, depth0.shape[0] - 1).long()
    d0 = depth0[vs, us]  # (N,)

    # Back-project: P0_cam = K^{-1} @ [u, v, 1] * d
    K_inv = torch.inverse(K)
    ones = torch.ones_like(query_xy[:, :1])
    p_homo = torch.cat([query_xy, ones], dim=-1)  # (N, 3)
    rays = p_homo @ K_inv.T                        # (N, 3)
    P0 = rays * d0.unsqueeze(-1)                   # (N, 3) in frame-0 camera

    # For each frame i: transform to world via T_0, then into frame i
    T0 = poses[0]
    predicted = torch.zeros(T, N, 2, device=device)
    predicted[0] = query_xy  # trivially correct on frame 0

    P_world = P0 @ T0[:3, :3].T + T0[:3, 3]        # (N, 3) world coords
    for i in range(1, T):
        Ti = poses[i]
        Ti_inv = torch.inverse(Ti)
        Pi = P_world @ Ti_inv[:3, :3].T + Ti_inv[:3, 3]  # (N, 3)
        z = Pi[:, 2].clamp(min=1e-6)
        u = (Pi[:, 0] * K[0, 0] / z) + K[0, 2]
        v = (Pi[:, 1] * K[1, 1] / z) + K[1, 2]
        predicted[i, :, 0] = u
        predicted[i, :, 1] = v
    return predicted


@torch.no_grad()
def _run_cotracker(
    model,
    frames: list[_TrackFrame],
    query_xy: torch.Tensor,      # (N, 2)
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CoTracker on the buffered frames with queries seeded on frame 0.

    Returns
    -------
    pred_tracks : (T, N, 2)   observed (u, v) per frame per point
    pred_vis    : (T, N)      float visibility / confidence in [0, 1]
    """
    T = len(frames)
    video = torch.stack([f.rgb for f in frames], dim=0)  # (T, 3, H, W) float 0-255
    video = video.unsqueeze(0).to(device)                 # (1, T, 3, H, W)
    # queries: (1, N, 3) as [t, x, y]. t=0 seeds on the oldest frame.
    query_xy = query_xy.to(device)
    queries = torch.cat([
        torch.zeros(query_xy.shape[0], 1, device=device),
        query_xy,
    ], dim=-1).unsqueeze(0)
    pred_tracks, pred_vis = model(video, queries=queries)
    # Shapes: (1, T, N, 2), (1, T, N)
    return pred_tracks[0], pred_vis[0]


@torch.no_grad()
def compute_pips_dynamic_mask(
    buffer: PointTrackBuffer,
    K: torch.Tensor,
    model,
    grid: int = 20,
    residual_thresh_px: float = 5.0,
    dilation_kernel: int = 9,
    current_depth: Optional[torch.Tensor] = None,
    semantic_mask: Optional[torch.Tensor] = None,
) -> Optional[tuple[torch.Tensor, dict]]:
    """Compute a dense dynamic mask using CoTracker trajectories.

    Returns
    -------
    mask : (H, W) float tensor in {0.0, 1.0}, 1 = STATIC (tracker's
        `mask=...` convention), 0 = DYNAMIC. None if buffer isn't full.
    info : diagnostic dict — pips_dynamic_pct (signal-only) + added_over_yolo_pct.
    """
    if not buffer.is_full():
        return None
    frames = buffer.frames()
    T = len(frames)
    H, W = frames[-1].depth.shape
    device = K.device

    # 1. Query grid on the OLDEST frame.
    query_xy = _make_query_grid(H, W, grid).to(device)  # (N, 2)
    depth0 = frames[0].depth

    # 2. Observed trajectories from CoTracker.
    pred_tracks, pred_vis = _run_cotracker(model, frames, query_xy, device)
    # pred_tracks: (T, N, 2); pred_vis: (T, N)

    # 3. Predicted-static positions.
    poses = [f.pose for f in frames]
    pred_static = _predict_static_positions(query_xy, depth0, poses, K)

    # 4. Per-frame residual.
    residual = (pred_tracks - pred_static).norm(dim=-1)  # (T, N)

    # Drop invalid seed depths (0) and invisibility — those are not reliable.
    us = query_xy[:, 0].clamp(0, W - 1).long()
    vs = query_xy[:, 1].clamp(0, H - 1).long()
    valid_seed_depth = depth0[vs, us] > 0  # (N,)
    valid = valid_seed_depth.unsqueeze(0) & (pred_vis > 0.5)  # (T, N)

    # 5. Median-correct to remove camera-egomotion bias (the DC component
    # of residual that is shared across all static points due to pose noise).
    # Subtract the per-frame median residual vector from each frame's residuals.
    # Compute separately in u and v to handle directional biases.
    delta_uv = pred_tracks - pred_static        # (T, N, 2)
    # Median across the valid subset per frame.
    med = torch.zeros(T, 2, device=device)
    for t in range(T):
        v_t = valid[t]
        if v_t.sum() > 0:
            med[t] = delta_uv[t, v_t].median(dim=0).values
    delta_corr = delta_uv - med.unsqueeze(1)     # (T, N, 2)
    residual = delta_corr.norm(dim=-1)           # (T, N)
    residual = residual.masked_fill(~valid, 0.0)

    # 6. Aggregate via max-over-time. A point that looks dynamic on any frame
    # of the window is classified dynamic.
    max_res = residual.max(dim=0).values        # (N,)

    # 7. Classify.
    dyn_pts = max_res > residual_thresh_px      # (N,)

    # 8. Scatter to a grid image then upsample + dilate.
    #    Grid is row-major across (y, x), so reshape N = grid*grid to (grid, grid).
    dyn_grid = dyn_pts.view(grid, grid).float().unsqueeze(0).unsqueeze(0)  # (1,1,g,g)
    dyn_dense = F.interpolate(dyn_grid, size=(H, W), mode="nearest")       # (1,1,H,W)
    # Dilate so a dynamic query covers its Voronoi-ish neighborhood.
    if dilation_kernel > 1:
        dyn_dense = F.max_pool2d(
            dyn_dense, kernel_size=dilation_kernel,
            stride=1, padding=dilation_kernel // 2,
        )
    dynamic_bool = dyn_dense.squeeze(0).squeeze(0) > 0.5  # (H, W) bool

    info: dict = {
        "pips_dynamic_pct": 100.0 * dynamic_bool.float().mean().item(),
        "pips_dyn_points": int(dyn_pts.sum().item()),
        "pips_total_points": int(valid[-1].sum().item()),
        "pips_median_residual_px": float(residual[valid].median().item()) if valid.any() else 0.0,
        "pips_max_residual_px": float(residual.max().item()),
    }

    # 9. UNION with semantic (YOLO) mask if present.
    # Incoming convention: semantic_mask 1=static, 0=dynamic.
    if semantic_mask is not None:
        sm = semantic_mask.to(device)
        if sm.dtype != torch.bool:
            sm = sm > 0.5
        semantic_dynamic = ~sm
        fused_dynamic = dynamic_bool | semantic_dynamic
        info["pips_added_over_yolo_pct"] = 100.0 * (
            (dynamic_bool & ~semantic_dynamic).float().mean().item()
        )
    else:
        fused_dynamic = dynamic_bool
        info["pips_added_over_yolo_pct"] = info["pips_dynamic_pct"]

    # Return static float mask (tracker convention: 1 = keep).
    static_mask = (~fused_dynamic).float()
    return static_mask, info
