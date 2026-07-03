"""Comprehensive d=16 re-run for Tab. 1 (3 scenes x 3 seeds, FULL mode)
and Tab. 3 (H1 in 3 modes: static / dynamic / full).

Saves trajectory NPZs to results/figures/ in the same naming convention
as scripts/export_trajectories.py so figures_r1_r2.py picks them up
without modification:

    traj_h{1,2,3}_<seqname>_seed{42,123,2024}.npz

Also writes a summary JSON to results/d16_rerun_summary.json with the
ATE-RMSE / gaussians / time per run, for paper Tab.~1 and Tab.~3.
"""
import sys
import os
import time
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

from dynlang_slam.utils.config import load_config
from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
from dynlang_slam.data.replica import ReplicaDataset, get_replica_intrinsics
from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.slam.pipeline import SLAMPipeline


SCENES = [
    dict(tag="h1", name="person_tracking", kind="bonn",
         subdir="rgbd_bonn_person_tracking", n_frames=100, stride=2),
    dict(tag="h2", name="balloon", kind="bonn",
         subdir="rgbd_bonn_balloon", n_frames=100, stride=2),
    dict(tag="h3", name="replica_room0", kind="replica",
         subdir="room0", n_frames=100, stride=1),
]
SEEDS = [42, 123, 2024]
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


def _seed_everything(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_cfg(scene, mode):
    """mode in {static, dynamic, full}."""
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs",
                                   "default.yaml"), [])

    if scene["kind"] == "bonn":
        cfg.dataset.type = "tum"
        cfg.dataset.image_height = 480
        cfg.dataset.image_width = 640
        cfg.dataset.depth_scale = 5000.0
    else:
        cfg.dataset.type = "replica"
        cfg.dataset.image_height = 680
        cfg.dataset.image_width = 1200
        cfg.dataset.depth_scale = 6553.5

    dyn_on  = mode in ("dynamic", "full")
    lang_on = (mode == "full")

    cfg.dynamic.enabled = dyn_on
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.language.enabled = lang_on
    if lang_on:
        cfg.language.extract_every_n = 2
        cfg.language.scales = ["whole"]
        cfg.language.sam_checkpoint = os.path.join(
            PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
        # d=16 is now the default in default.yaml; verify it's set:
        cfg.gaussians.lang_feat_dim = 16
        cfg.language.autoencoder.latent_dim = 16
        cfg.language.autoencoder.warmup_frames = 20
    return cfg


def build_dataset(scene):
    if scene["kind"] == "bonn":
        return TUMDataset(
            data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", scene["subdir"]),
            height=480, width=640, depth_scale=5000.0,
            max_frames=scene["n_frames"], stride=scene["stride"],
        )
    return ReplicaDataset(
        data_dir=os.path.join(PROJECT_ROOT, "data", "Replica", scene["subdir"]),
        height=680, width=1200, depth_scale=6553.5,
        max_frames=scene["n_frames"], stride=scene["stride"],
    )


def run_one(scene, mode, seed, save_traj=False):
    print("=" * 60)
    print(f"d=16 re-run: scene={scene['tag']} mode={mode} seed={seed}")
    print("=" * 60)
    _seed_everything(seed)

    cfg = build_cfg(scene, mode)
    cfg.training.seed = seed
    intrinsics = (get_bonn_intrinsics() if scene["kind"] == "bonn"
                  else get_replica_intrinsics())
    dataset = build_dataset(scene)

    device = "cuda"
    gmap = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intrinsics, device=device)

    t_start = time.time()
    slam.process_first_frame(gmap, dataset[0])
    for i in range(1, len(dataset)):
        info = slam.process_frame(gmap, dataset[i], use_gt_pose=False)
        if i % 25 == 0 or i == len(dataset) - 1:
            print(f"  Frame {i:3d}/{len(dataset)-1} | "
                  f"ATE={info['ate']*100:.2f}cm | G={info['total_gaussians']}")
    total_time = time.time() - t_start

    gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]
    ate_rmse_cm = slam.compute_ate_rmse(gt_poses) * 100

    result = dict(
        scene=scene["tag"], mode=mode, seed=seed,
        n_frames=len(dataset),
        ate_rmse_cm=float(ate_rmse_cm),
        gaussians=int(gmap.num_gaussians),
        total_time_s=float(total_time),
    )

    # Save trajectory NPZ for figures_r1_r2.py
    if save_traj:
        est_c2w = np.stack([slam.estimated_poses[i].cpu().numpy()
                            for i in range(len(dataset))], axis=0)
        gt_c2w = np.stack([dataset[i]["pose"].cpu().numpy()
                           for i in range(len(dataset))], axis=0)
        out = os.path.join(
            FIG_DIR, f"traj_{scene['tag']}_{scene['name']}_seed{seed}.npz")
        np.savez(out, est_c2w=est_c2w, gt_c2w=gt_c2w,
                 frame_ids=np.arange(len(dataset)),
                 scene=scene["tag"], sequence=scene["name"],
                 seed=seed, runtime_s=total_time)
        print(f"  saved {out}")
        result["traj_path"] = out

    print(f"  -> ATE={ate_rmse_cm:.2f}cm  G={gmap.num_gaussians}  "
          f"time={total_time:.0f}s")

    del gmap, slam
    torch.cuda.empty_cache()
    return result


def main():
    all_results = dict(tab1=[], tab3=[])

    # ---- Tab 1: 3 scenes x 3 seeds, FULL mode ----
    for scene in SCENES:
        for seed in SEEDS:
            try:
                r = run_one(scene, mode="full", seed=seed, save_traj=True)
                all_results["tab1"].append(r)
            except Exception as e:
                print(f"  FAILED scene={scene['tag']} seed={seed}: "
                      f"{type(e).__name__}: {e}")
                all_results["tab1"].append(dict(
                    scene=scene["tag"], seed=seed, error=str(e)))

    # ---- Tab 3: H1 in 3 modes, seed 42 ----
    h1 = SCENES[0]
    for mode in ("static", "dynamic", "full"):
        try:
            r = run_one(h1, mode=mode, seed=42, save_traj=False)
            all_results["tab3"].append(r)
        except Exception as e:
            print(f"  FAILED H1 mode={mode}: {type(e).__name__}: {e}")
            all_results["tab3"].append(dict(
                mode=mode, error=str(e)))

    # ---- Save JSON summary ----
    out_json = os.path.join(PROJECT_ROOT, "results", "d16_rerun_summary.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== SUMMARY written to {out_json} ===")

    # Pretty print Tab 1 stats
    print("\n=== Tab. 1: multi-seed ATE-RMSE @ d=16 ===")
    by_scene = {}
    for r in all_results["tab1"]:
        if "error" in r:
            continue
        by_scene.setdefault(r["scene"], []).append(r["ate_rmse_cm"])
    for tag, vals in by_scene.items():
        m = float(np.mean(vals)); s = float(np.std(vals))
        print(f"  {tag}: ATE = {m:.2f} +/- {s:.2f} cm  (n={len(vals)} seeds)")

    print("\n=== Tab. 3: H1 three-mode ATE @ d=16 ===")
    for r in all_results["tab3"]:
        if "error" in r:
            continue
        print(f"  {r['mode']:>8}: ATE = {r['ate_rmse_cm']:.2f} cm  "
              f"G={r['gaussians']}  time={r['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
