"""Smoke test: load CoTracker, track a grid of points through BONN frames.

Goal: verify that CoTracker runs on our hardware, produces sensible tracks,
and gives us the signal we need for dynamic-mask generation.

Expected output:
  - Pred tracks of shape (1, T, N, 2) -- (u,v) per point per frame
  - Visibility of shape (1, T, N) -- binary occlusion flag
"""
import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from dynlang_slam.data.tum import TUMDataset

SEQ = os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_balloon")
T = 8  # window length
GRID = 20  # GRID x GRID query points = 400 queries

device = "cuda"

# Load 8 frames
ds = TUMDataset(data_dir=SEQ, height=480, width=640, depth_scale=5000.0,
                max_frames=T, stride=2)
frames = []
for i in range(T):
    rgb = ds[i]["rgb"]  # (3, H, W) or (H, W, 3) — check
    if rgb.dim() == 3 and rgb.shape[0] == 3:
        rgb = rgb.permute(1, 2, 0)  # (H, W, 3)
    frames.append(rgb)
video = torch.stack(frames, dim=0).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
if video.max() <= 1.0:
    video = video * 255.0
video = video.unsqueeze(0).to(device)  # (1, T, 3, H, W)
print(f"Video tensor: {tuple(video.shape)} dtype={video.dtype} range=[{video.min().item():.1f}, {video.max().item():.1f}]")

# Build grid of query points on frame 0: (N, 3) = [t, x, y]
H, W = 480, 640
ys = torch.linspace(10, H - 10, GRID)
xs = torch.linspace(10, W - 10, GRID)
vv, uu = torch.meshgrid(ys, xs, indexing="ij")
queries_2d = torch.stack([uu.flatten(), vv.flatten()], dim=-1)  # (N, 2)
queries = torch.cat([torch.zeros(queries_2d.shape[0], 1), queries_2d], dim=-1)  # (N, 3)
queries = queries.unsqueeze(0).to(device)  # (1, N, 3)
print(f"Queries: {tuple(queries.shape)}  (N={queries.shape[1]})")

# Offline CoTracker: one-shot over the clip
model = torch.hub.load("facebookresearch/co-tracker", "cotracker2",
                       trust_repo=True).to(device).eval()
print(f"Model: {type(model).__name__}")

t0 = time.time()
with torch.no_grad():
    pred_tracks, pred_vis = model(video, queries=queries)
print(f"Inference time: {(time.time()-t0)*1000:.0f}ms")
print(f"pred_tracks: {tuple(pred_tracks.shape)}  pred_vis: {tuple(pred_vis.shape)}")
print(f"Track displacement stats (frame 0 -> frame T-1):")
d = (pred_tracks[0, -1] - pred_tracks[0, 0]).norm(dim=-1)  # (N,)
print(f"  min={d.min().item():.1f}px  max={d.max().item():.1f}px  mean={d.mean().item():.1f}px  median={d.median().item():.1f}px")

# Visibility stats
vis_sum = pred_vis[0].sum(dim=0)  # (N,)
print(f"Visible frames per point: min={vis_sum.min().item()}  mean={vis_sum.float().mean().item():.1f}  max={vis_sum.max().item()}")

# Memory
if torch.cuda.is_available():
    print(f"GPU memory after inference: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB peak")
