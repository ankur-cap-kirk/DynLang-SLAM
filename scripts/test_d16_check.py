"""Quick test: does d=16 latent_dim recover with the new
warmup=20, threshold=0.70 settings?

Runs only the FULL mode (dynamic + language) on H1 BONN
person_tracking, 100 frames. Reports:
  - Whether the autoencoder froze
  - At what frame
  - Final language feature non-zero %
  - Per-query mean relevancy (a featureless heatmap = mean ≈ 0.5)

If AE freezes and lang non-zero % > 80% and per-query mean
relevancy spreads away from 0.5 → d=16 is recoverable.
"""
import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

SEQUENCE = os.path.join(PROJECT_ROOT, "data", "BONN",
                        "rgbd_bonn_person_tracking")
N_FRAMES = 100
STRIDE = 2
TARGET_DIM = 16

QUERIES = ["person", "chair", "monitor", "floor", "desk"]


def main():
    print("=" * 60)
    print(f"d={TARGET_DIM} viability test: H1 person_tracking, "
          f"{N_FRAMES} frames, FULL mode")
    print("=" * 60)

    cfg = load_config(os.path.join(PROJECT_ROOT, "configs",
                                   "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0

    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"

    cfg.language.enabled = True
    cfg.language.extract_every_n = 2
    cfg.language.scales = ["whole"]
    cfg.language.sam_checkpoint = os.path.join(
        PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")

    # The two changes under test:
    cfg.gaussians.lang_feat_dim = TARGET_DIM
    cfg.language.autoencoder.latent_dim = TARGET_DIM
    # Keep the other parts of the bug fix:
    cfg.language.autoencoder.warmup_frames = 20

    intrinsics = get_bonn_intrinsics()
    dataset = TUMDataset(
        data_dir=SEQUENCE,
        height=480, width=640,
        depth_scale=5000.0,
        max_frames=N_FRAMES,
        stride=STRIDE,
    )

    device = "cuda"
    gaussian_map = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)

    t_start = time.time()
    slam.process_first_frame(gaussian_map, dataset[0])

    ae_frozen_at = None
    for i in range(1, len(dataset)):
        info = slam.process_frame(gaussian_map, dataset[i],
                                  use_gt_pose=False)
        if (ae_frozen_at is None and slam._autoencoder is not None
                and getattr(slam._autoencoder, "is_frozen", False)):
            ae_frozen_at = i
            print(f"  >>> AE FROZE at frame {i}")

        if i % 10 == 0 or i == len(dataset) - 1:
            parts = [f"Frame {i:3d}/{len(dataset)-1}",
                     f"ATE={info['ate']*100:.2f}cm",
                     f"G={info['total_gaussians']}"]
            if "lang_loss" in info:
                parts.append(f"lang_loss={info['lang_loss']:.4f}")
            if "lang_extract_time" in info:
                parts.append(f"lang_extract={info['lang_extract_time']:.1f}s")
            print(f"  {' | '.join(parts)}")

    total_time = time.time() - t_start
    gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    ate_rmse = slam.compute_ate_rmse(gt_poses) * 100

    lang_feats = gaussian_map.lang_feats.data
    nonzero = (lang_feats.abs().sum(dim=-1) > 0.01).sum().item()
    nonzero_pct = nonzero / lang_feats.shape[0] * 100

    print(f"\n=== RESULTS at d={TARGET_DIM} ===")
    print(f"  ATE-RMSE:           {ate_rmse:.2f} cm")
    print(f"  Total Gaussians:    {gaussian_map.num_gaussians}")
    print(f"  AE frozen:          {ae_frozen_at is not None} "
          f"(at frame {ae_frozen_at})")
    print(f"  Lang feat non-zero: {nonzero}/{lang_feats.shape[0]} "
          f"({nonzero_pct:.1f}%)")
    print(f"  Lang norm range:    "
          f"[{lang_feats.norm(dim=-1).min().item():.4f}, "
          f"{lang_feats.norm(dim=-1).max().item():.4f}]")
    print(f"  Total time:         {total_time:.1f}s "
          f"({total_time/len(dataset):.2f}s/frame)")

    if (slam._lang_initialized and slam._autoencoder is not None
            and getattr(slam._autoencoder, "is_frozen", False)):
        print(f"\n=== Per-query relevancy stats (top-50 Gaussians) ===")
        print("  query        rel_min   rel_max   rel_mean")
        for q in QUERIES:
            try:
                r = slam.query_3d(gaussian_map, q, top_k=50,
                                  use_relevancy=True)
                s = r["top_k_scores"]
                if torch.is_tensor(s):
                    s = s.cpu().numpy()
                print(f"  {q:<11}  "
                      f"{float(np.min(s)):>7.3f}   "
                      f"{float(np.max(s)):>7.3f}   "
                      f"{float(np.mean(s)):>7.3f}")
            except Exception as e:
                print(f"  {q:<11}  query failed: {e}")

    print("\n=== VERDICT ===")
    if ae_frozen_at is None:
        print("  FAIL: AE never froze. d=16 is NOT recoverable.")
    elif nonzero_pct < 50:
        print("  FAIL: AE froze but feature coverage too low "
              "(<50%). d=16 is NOT recoverable.")
    else:
        print("  PASS: AE froze, features populated. "
              "d=16 IS recoverable. Paper can be updated.")


if __name__ == "__main__":
    main()
