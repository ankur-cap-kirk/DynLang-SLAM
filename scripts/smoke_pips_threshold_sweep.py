"""Sweep PIPs residual threshold to find the separation between noise and
real dynamic motion on BONN balloon. We use GT pose (so pose error isn't a
confounder) and sweep thresh_px over a sensible range.

Expected: at low thresholds PIPs flags most of the image (noise floor);
at high thresholds, only the balloon. Find the 'knee' where dyn% stabilizes.
"""
import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch, numpy as np
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.dynamic import PointTrackBuffer, compute_pips_dynamic_mask

device = "cuda"
SEQ = os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_balloon")
T_WIN = 8
N_FRAMES = 20
STRIDE = 2

ds = TUMDataset(data_dir=SEQ, height=480, width=640, depth_scale=5000.0,
                max_frames=N_FRAMES, stride=STRIDE)
K = get_bonn_intrinsics()["K"].to(device)
model = torch.hub.load("facebookresearch/co-tracker", "cotracker2",
                       trust_repo=True).to(device).eval()

buffer = PointTrackBuffer(window_size=T_WIN)
for i in range(N_FRAMES):
    f = ds[i]
    buffer.push(f["rgb"].to(device), f["depth"].to(device), f["pose"].to(device))

# Re-use the last full window. Call compute at several thresholds and
# look at how dyn% responds.
thresholds = [5, 10, 15, 20, 25, 30, 40, 50, 75]
print(f"Threshold sweep on BONN balloon at frame {N_FRAMES-1}:")
print(f"{'thresh_px':>10} {'pips_dyn%':>10} {'n_dyn_pts':>10} {'n_total':>10}")
for th in thresholds:
    result = compute_pips_dynamic_mask(
        buffer=buffer, K=K, model=model,
        grid=20, residual_thresh_px=th, semantic_mask=None,
    )
    mask, info = result
    print(f"{th:10.1f} {info['pips_dynamic_pct']:10.2f} "
          f"{info['pips_dyn_points']:10d} {info['pips_total_points']:10d}")

# Also dump the full residual distribution at max-over-time aggregation
# so we can read off the noise-vs-signal split visually.
print("\nResidual distribution (percentiles):")
# Re-run once internally to get raw residuals — simpler: recompute inline
from dynlang_slam.dynamic.pips_mask import (
    _make_query_grid, _run_cotracker, _predict_static_positions,
)
query_xy = _make_query_grid(480, 640, 20).to(device)
frames = buffer.frames()
pred_tracks, pred_vis = _run_cotracker(model, frames, query_xy, device)
poses = [f.pose for f in frames]
pred_static = _predict_static_positions(query_xy, frames[0].depth, poses, K)
delta_uv = pred_tracks - pred_static
# median-correct per frame
T = len(frames)
med = torch.zeros(T, 2, device=device)
for t in range(T):
    med[t] = delta_uv[t].median(dim=0).values
residual = (delta_uv - med.unsqueeze(1)).norm(dim=-1)
max_res = residual.max(dim=0).values.cpu().numpy()
for p in [10, 25, 50, 75, 85, 90, 95, 99]:
    print(f"  p{p:>3}: {np.percentile(max_res, p):6.1f} px")
print(f"  max : {max_res.max():6.1f} px")
