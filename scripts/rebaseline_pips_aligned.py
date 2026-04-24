"""Re-check PIPs (CoTracker class-agnostic mask) under Umeyama-aligned ATE.

Under the raw metric, Bp_pips was rejected:
    H1 person_tracking (thresh=25): 19.83 vs baseline 20.32  (~0.5 cm improvement)
    H2 balloon         (thresh=25): 57.69 vs baseline 57.35  (~0.3 cm regression)

Both were within noise, so the intervention was rejected for not moving
the needle. Prediction under the aligned metric (where baselines are
H1=7.56, H2=8.55): PIPs should also move the needle by < 0.5 cm because
the dominant residual error is a rigid offset that a mapping-side mask
cannot fix. This script verifies.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline


N_FRAMES = int(os.environ.get("DYNLANG_N_FRAMES_BONN", "100"))
STRIDE = int(os.environ.get("DYNLANG_STRIDE_BONN", "2"))
THRESH = float(os.environ.get("DYNLANG_PIPS_THRESH", "25.0"))  # protocol-permitted


# Aligned baselines from earlier re-baseline run (no PIPs, no language,
# dynamic ON, yolov8n-seg, 100f stride=2).
BASELINE_ALIGNED = {
    "rgbd_bonn_person_tracking": 7.56,
    "rgbd_bonn_balloon":         8.55,
}
BASELINE_RAW = {
    "rgbd_bonn_person_tracking": 29.16,
    "rgbd_bonn_balloon":         57.89,
}


def run(seq_name: str):
    print(f"\n{'='*64}")
    print(f" BONN {seq_name}   pips.enabled=True  thresh={THRESH}")
    print(f"{'='*64}")
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = True
    cfg.dynamic.pips.thresh_px = THRESH
    cfg.language.enabled = False

    dataset = TUMDataset(
        data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", seq_name),
        height=480, width=640, depth_scale=5000.0,
        max_frames=N_FRAMES, stride=STRIDE,
    )
    device = "cuda"
    gmap = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=get_bonn_intrinsics(), device=device)
    slam.process_first_frame(gmap, dataset[0])

    t0 = time.time()
    for i in range(1, len(dataset)):
        slam.process_frame(gmap, dataset[i], use_gt_pose=False)
    gt = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    raw = slam.compute_ate_rmse(gt, align=False) * 100
    aln = slam.compute_ate_rmse(gt, align=True)  * 100
    dt = time.time() - t0

    delta_raw = raw - BASELINE_RAW[seq_name]
    delta_aln = aln - BASELINE_ALIGNED[seq_name]
    print(f" runtime={dt:.1f}s")
    print(f" raw     ATE-RMSE: {raw:.2f} cm   (baseline {BASELINE_RAW[seq_name]:.2f}, "
          f"delta {delta_raw:+.2f})")
    print(f" ALIGNED ATE-RMSE: {aln:.2f} cm   (baseline {BASELINE_ALIGNED[seq_name]:.2f}, "
          f"delta {delta_aln:+.2f})")
    del gmap, slam
    torch.cuda.empty_cache()
    return raw, aln, delta_raw, delta_aln


results = {}
for seq in ["rgbd_bonn_person_tracking", "rgbd_bonn_balloon"]:
    results[seq] = run(seq)

print(f"\n{'='*64}")
print(f" PIPs RE-CHECK SUMMARY (aligned ATE)")
print(f"{'='*64}")
print(f" {'scene':<28s} {'raw':>10s} {'aligned':>10s} {'d_raw':>10s} {'d_aligned':>12s}")
print(f" {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
for seq, (raw, aln, d_raw, d_aln) in results.items():
    short = seq.replace("rgbd_bonn_", "")
    print(f" {short:<28s} {raw:>10.2f} {aln:>10.2f} {d_raw:>+10.2f} {d_aln:>+12.2f}")

# Simple heuristic verdict
d_h1 = results["rgbd_bonn_person_tracking"][3]
d_h2 = results["rgbd_bonn_balloon"][3]
print()
if d_h1 < -0.5 or d_h2 < -1.0:
    print(" VERDICT: PIPs shows meaningful aligned-ATE improvement on at least one")
    print("          scene. Re-adjudicate Bp intervention under aligned metric.")
elif abs(d_h1) < 0.5 and abs(d_h2) < 1.0:
    print(" VERDICT: PIPs within noise on both scenes under aligned ATE too.")
    print("          Prediction confirmed: Bp rejection stands.")
    print("          Move on to bootstrap refinement to attack rigid offset.")
else:
    print(" VERDICT: mixed/regressive. PIPs does not help under aligned ATE.")
