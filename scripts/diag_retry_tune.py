"""Protocol-permitted tuning of retry_loss_ratio_thresh.

Permitted (from protocol.md): thresh ∈ {2.0, 3.0}, num_hypotheses ∈ {3, 6}.
This diag lets each knob be overridden via env vars so we can cheaply
exhaust the tuning budget before declaring REJECT.
"""
import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline

SEQ = os.environ.get("DYNLANG_SEQUENCE", "rgbd_bonn_person_tracking")
THRESH = float(os.environ.get("DYNLANG_RETRY_THRESH", "2.0"))
NHYP = int(os.environ.get("DYNLANG_RETRY_NHYP", "4"))
N_FRAMES = int(os.environ.get("DYNLANG_N_FRAMES", "100"))

print(f"== tune: seq={SEQ} thresh={THRESH} nhyp={NHYP} ==")

cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
cfg.dataset.type = "tum"
cfg.dataset.image_height = 480
cfg.dataset.image_width = 640
cfg.dataset.depth_scale = 5000.0
cfg.dynamic.enabled = True
cfg.dynamic.yolo_model = "yolov8n-seg"
cfg.language.enabled = False
cfg.slam.tracking.retry_loss_ratio_thresh = THRESH
cfg.slam.tracking.retry_num_hypotheses = NHYP

dataset = TUMDataset(
    data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", SEQ),
    height=480, width=640, depth_scale=5000.0,
    max_frames=N_FRAMES, stride=2,
)

device = "cuda"
gaussian_map = GaussianMap(
    sh_degree=cfg.gaussians.sh_degree,
    lang_feat_dim=cfg.gaussians.lang_feat_dim,
    init_opacity=cfg.gaussians.init_opacity,
    device=device,
)
slam = SLAMPipeline(cfg=cfg, intrinsics=get_bonn_intrinsics(), device=device)
slam.process_first_frame(gaussian_map, dataset[0])

retry_n = 0
losses = []
t0 = time.time()
for i in range(1, len(dataset)):
    info = slam.process_frame(gaussian_map, dataset[i], use_gt_pose=False)
    losses.append(info.get("tracking_loss", 0))
    if info.get("retry_fired", False):
        retry_n += 1
        print(f"  [RETRY] f={i} trig={info.get('retry_trigger_ratio', 0):.2f}x win={info.get('retry_winner')}")

gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
ate = slam.compute_ate_rmse(gt_poses) * 100
arr = np.array(losses)
print(f"ATE={ate:.2f}cm | retries={retry_n}/{len(dataset)-1} | "
      f"loss med={np.median(arr):.4f} max={arr.max():.4f} max/med={arr.max()/np.median(arr):.2f}x | "
      f"{time.time()-t0:.1f}s")
