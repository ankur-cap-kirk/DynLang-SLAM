"""BR1 evaluation: bootstrap-window joint pose refinement under aligned ATE.

Pre-registered protocol: research/experiments/bootstrap-refinement/protocol.md

Baselines (from aligned_baselines_2026_04_23):
    H1 rgbd_bonn_person_tracking: 7.56 cm
    H2 rgbd_bonn_balloon:         8.55 cm
    H3 replica_room0:             3.68 cm

Accept budgets (aligned ATE-RMSE):
    H1 <= 5.50 cm   H2 <= 6.00 cm   H3 <= 4.10 cm

Run with defaults:
    python scripts/br1_eval.py
Select subset:
    DYNLANG_SCENES=h1,h2 python scripts/br1_eval.py
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
from pathlib import Path

from dynlang_slam.utils.config import load_config
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline


SCENES = os.environ.get("DYNLANG_SCENES", "h1,h2,h3").split(",")
N_BONN = int(os.environ.get("DYNLANG_N_FRAMES_BONN", "100"))
STRIDE_BONN = int(os.environ.get("DYNLANG_STRIDE_BONN", "2"))
N_REPLICA = int(os.environ.get("DYNLANG_N_FRAMES_REPLICA", "100"))

# Bootstrap knobs (protocol-fixed — not for tuning)
BS_N = int(os.environ.get("DYNLANG_BS_N", "15"))
BS_ITERS = int(os.environ.get("DYNLANG_BS_ITERS", "200"))

BASELINE_ALIGNED = {
    "rgbd_bonn_person_tracking": 7.56,
    "rgbd_bonn_balloon":         8.55,
    "replica_room0":             3.68,
}
BUDGET_ALIGNED = {
    "rgbd_bonn_person_tracking": 5.50,
    "rgbd_bonn_balloon":         6.00,
    "replica_room0":             4.10,
}


def _build_cfg_bonn(bootstrap_on: bool):
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = False  # BR1 tested WITHOUT pips (per protocol)
    cfg.language.enabled = False
    # Bootstrap knobs
    cfg.slam.bootstrap.enabled = bootstrap_on
    cfg.slam.bootstrap.n_frames = BS_N
    cfg.slam.bootstrap.iterations = BS_ITERS
    return cfg


def _build_cfg_replica(bootstrap_on: bool):
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.scene = "room0"
    cfg.dataset.max_frames = N_REPLICA
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.dynamic.pips.enabled = False
    cfg.language.enabled = False
    cfg.slam.bootstrap.enabled = bootstrap_on
    cfg.slam.bootstrap.n_frames = BS_N
    cfg.slam.bootstrap.iterations = BS_ITERS
    return cfg


def run_bonn(label: str, seq_name: str, bootstrap_on: bool):
    from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
    tag = "BR1-ON" if bootstrap_on else "BR1-OFF"
    print(f"\n{'='*72}")
    print(f" {label} {seq_name}  bootstrap={tag}  n={N_BONN} stride={STRIDE_BONN}")
    print(f"{'='*72}")

    cfg = _build_cfg_bonn(bootstrap_on)
    dataset = TUMDataset(
        data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", seq_name),
        height=480, width=640, depth_scale=5000.0,
        max_frames=N_BONN, stride=STRIDE_BONN,
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
    bs_diag = None
    for i in range(1, len(dataset)):
        slam.process_frame(gmap, dataset[i], use_gt_pose=False)
        if bs_diag is None:
            bs_diag = slam.maybe_run_bootstrap(gmap)
            if bs_diag is not None and bs_diag.get("bootstrap_fired"):
                print(f"  [BR1 fired at frame {slam.frame_count}] "
                      f"K_refined={bs_diag['K_refined']} "
                      f"max_Δt={bs_diag['pose_trans_max_m']*100:.2f}cm "
                      f"mean_Δt={bs_diag['pose_trans_mean_m']*100:.2f}cm "
                      f"final_loss={bs_diag['final_loss']:.4f}", flush=True)

    gt = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    raw = slam.compute_ate_rmse(gt, align=False) * 100
    aln = slam.compute_ate_rmse(gt, align=True) * 100
    dt = time.time() - t0

    baseline = BASELINE_ALIGNED[seq_name]
    budget = BUDGET_ALIGNED[seq_name]
    delta = aln - baseline
    print(f" runtime={dt:.1f}s")
    print(f" raw     ATE-RMSE: {raw:.2f} cm")
    print(f" ALIGNED ATE-RMSE: {aln:.2f} cm   (baseline {baseline:.2f}, "
          f"delta {delta:+.2f}, budget <= {budget:.2f})")
    if bootstrap_on and bs_diag is not None and bs_diag.get("bootstrap_fired"):
        print(f" BR1 per-KF translation deltas (cm): "
              f"{[f'{d*100:.1f}' for d in bs_diag['pose_trans_deltas_m']]}")
        print(f" BR1 per-KF rotation    deltas (deg): "
              f"{[f'{d:.2f}' for d in bs_diag['pose_rot_deltas_deg']]}")

    del gmap, slam
    torch.cuda.empty_cache()
    return raw, aln, delta, bs_diag


def run_replica(label: str, bootstrap_on: bool):
    from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
    tag = "BR1-ON" if bootstrap_on else "BR1-OFF"
    print(f"\n{'='*72}")
    print(f" {label} replica_room0  bootstrap={tag}  n={N_REPLICA}")
    print(f"{'='*72}")

    cfg = _build_cfg_replica(bootstrap_on)
    dataset_path = Path(PROJECT_ROOT) / cfg.dataset.path / "room0"
    dataset = ReplicaDataset(
        data_dir=str(dataset_path),
        height=cfg.dataset.image_height,
        width=cfg.dataset.image_width,
        depth_scale=cfg.dataset.depth_scale,
        max_frames=N_REPLICA,
    )
    intrinsics = get_replica_intrinsics(
        fx=cfg.camera.fx, fy=cfg.camera.fy,
        cx=cfg.camera.cx, cy=cfg.camera.cy,
        height=cfg.dataset.image_height, width=cfg.dataset.image_width,
    )
    device = "cuda"
    gmap = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)
    slam.process_first_frame(gmap, dataset[0])

    t0 = time.time()
    bs_diag = None
    for i in range(1, len(dataset)):
        slam.process_frame(gmap, dataset[i], use_gt_pose=False)
        if bs_diag is None:
            bs_diag = slam.maybe_run_bootstrap(gmap)
            if bs_diag is not None and bs_diag.get("bootstrap_fired"):
                print(f"  [BR1 fired at frame {slam.frame_count}] "
                      f"K_refined={bs_diag['K_refined']} "
                      f"max_Δt={bs_diag['pose_trans_max_m']*100:.2f}cm "
                      f"mean_Δt={bs_diag['pose_trans_mean_m']*100:.2f}cm "
                      f"final_loss={bs_diag['final_loss']:.4f}", flush=True)

    gt = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    raw = slam.compute_ate_rmse(gt, align=False) * 100
    aln = slam.compute_ate_rmse(gt, align=True) * 100
    dt = time.time() - t0

    baseline = BASELINE_ALIGNED["replica_room0"]
    budget = BUDGET_ALIGNED["replica_room0"]
    delta = aln - baseline
    print(f" runtime={dt:.1f}s")
    print(f" raw     ATE-RMSE: {raw:.2f} cm")
    print(f" ALIGNED ATE-RMSE: {aln:.2f} cm   (baseline {baseline:.2f}, "
          f"delta {delta:+.2f}, budget <= {budget:.2f})")
    if bootstrap_on and bs_diag is not None and bs_diag.get("bootstrap_fired"):
        print(f" BR1 per-KF translation deltas (cm): "
              f"{[f'{d*100:.1f}' for d in bs_diag['pose_trans_deltas_m']]}")

    del gmap, slam
    torch.cuda.empty_cache()
    return raw, aln, delta, bs_diag


# --- Run BR1-ON for all requested scenes ---
results = {}
if "h1" in SCENES:
    results["H1_person_tracking"] = run_bonn("H1", "rgbd_bonn_person_tracking", True)
if "h2" in SCENES:
    results["H2_balloon"] = run_bonn("H2", "rgbd_bonn_balloon", True)
if "h3" in SCENES:
    results["H3_replica_room0"] = run_replica("H3", True)


# --- Summary + accept/reject decision ---
print(f"\n{'='*72}\n BR1 SUMMARY — aligned ATE vs pre-registered budgets\n{'='*72}")
print(f" {'scene':<28s} {'aligned':>8s} {'baseline':>9s} {'delta':>8s} "
      f"{'budget':>8s} {'verdict':>10s}")
print(f" {'-'*28} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*10}")

accepts = {}
for name, (raw, aln, delta, _) in results.items():
    # Figure out which budget applies
    if "person_tracking" in name:
        key = "rgbd_bonn_person_tracking"
    elif "balloon" in name:
        key = "rgbd_bonn_balloon"
    else:
        key = "replica_room0"
    baseline = BASELINE_ALIGNED[key]
    budget = BUDGET_ALIGNED[key]
    passed = aln <= budget
    verdict = "PASS" if passed else "FAIL"
    accepts[key] = passed
    print(f" {name:<28s} {aln:>8.2f} {baseline:>9.2f} {delta:>+8.2f} "
          f"{budget:>8.2f} {verdict:>10s}")

print()
all_pass = all(accepts.values()) if accepts else False
h3_safety_ok = accepts.get("replica_room0", True)
h1_improves = (
    BASELINE_ALIGNED["rgbd_bonn_person_tracking"]
    - results.get("H1_person_tracking", (0, BASELINE_ALIGNED["rgbd_bonn_person_tracking"], 0, None))[1]
) >= 0.5 if "H1_person_tracking" in results else None

if all_pass:
    print(" VERDICT: ACCEPT — BR1 meets all pre-registered budgets.")
    print("          Update research-state.yaml; enable bootstrap by default.")
elif not h3_safety_ok:
    print(" VERDICT: REJECT — H3 safety regression. BR1 hurts static scenes.")
elif h1_improves is False:
    print(" VERDICT: REJECT — H1 improvement below 0.5 cm noise floor.")
else:
    print(" VERDICT: PARTIAL — see per-scene table above.")
    print("          Document mechanism in analysis.md; may keep gated.")

print(f"\n SOTA ref (aligned): DG-SLAM=4.73 cm, BDGS-SLAM=4.03 cm")
