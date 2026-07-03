"""Open-vocabulary 3D mIoU evaluation on semantic Replica (vMAP renders).

Protocol (plan E5, week 3):
  1. Run the full DynLang-SLAM pipeline on the vMAP Replica room_0 sequence
     (RGB-D + poses; semantic images are used ONLY for evaluation).
  2. Unproject every Nth frame's semantic labels with GT depth+pose into a
     labeled 3D point cloud.
  3. Label each Gaussian with the class of its nearest GT point (within a
     distance cap; unlabeled otherwise).
  4. Query the map with each class NAME through the open-vocabulary pathway
     (CLIP text -> autoencoder latent -> cosine vs per-Gaussian features);
     assign each Gaussian to its argmax class.
  5. Report per-class IoU and mIoU over Gaussians that are both GT-labeled
     and pass the feature norm gate.

Unlike closed-set semantic SLAM (SGS-SLAM et al.), classification here is
zero-shot from text — no semantic supervision enters the map.

Data: data/vmap_replica/room_0 (vMAP demo archive,
https://huggingface.co/datasets/kxic/vMAP). Renders are 1200x680,
depth uint16 in mm, semantic_class uint8 habitat class ids,
traj_w_c.txt = row-major 4x4 camera-to-world (OpenCV convention),
intrinsics fx=fy=600, cx=599.5, cy=339.5 (90 deg hfov).

Usage:
    python scripts/eval_miou_replica.py [--frames 100] [--stride 1]
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from PIL import Image

SEQ = os.path.join(PROJECT_ROOT, "data", "vmap_replica", "room_0", "imap", "00")
INFO = os.path.join(PROJECT_ROOT, "data", "vmap_replica", "room_0",
                    "habitat", "info_semantic.json")
DEPTH_SCALE = 1000.0
MIN_GT_POINTS = 500          # classes with fewer GT points are skipped
GT_DIST_CAP = 0.05           # meters: Gaussian->GT-point association cap


def load_frames(n_frames: int, stride: int):
    poses = np.loadtxt(os.path.join(SEQ, "traj_w_c.txt")).reshape(-1, 4, 4)
    frames = []
    for k in range(n_frames):
        i = k * stride
        rgb = np.asarray(Image.open(os.path.join(SEQ, "rgb", f"rgb_{i}.png")))
        depth = np.asarray(Image.open(
            os.path.join(SEQ, "depth", f"depth_{i}.png"))).astype(np.float32)
        sem = np.asarray(Image.open(os.path.join(
            SEQ, "semantic_class", f"semantic_class_{i}.png"))).astype(np.int64)
        frames.append({
            "frame_id": k,
            "rgb": torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0,
            "depth": torch.from_numpy(depth.copy()).unsqueeze(0) / DEPTH_SCALE,
            "pose": torch.from_numpy(poses[i].copy()).float(),
            "semantic": torch.from_numpy(sem.copy()),
        })
    return frames


def class_names():
    info = json.load(open(INFO))
    return {c["id"]: c["name"] for c in info["classes"]}


def build_gt_pointcloud(frames, K, label_every: int, device: str):
    pts_all, lab_all = [], []
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    for i, fr in enumerate(frames):
        if i % label_every:
            continue
        depth = fr["depth"].squeeze(0).to(device)
        sem = fr["semantic"].to(device)
        pose = fr["pose"].to(device)
        H, W = depth.shape
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij")
        valid = (depth > 0) & (sem > 0)
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        pts_cam = torch.stack([x, y, z], dim=-1)
        R, t = pose[:3, :3], pose[:3, 3]
        pts_all.append((pts_cam @ R.T + t)[::8])
        lab_all.append(sem[valid][::8])
    return torch.cat(pts_all), torch.cat(lab_all)


def label_gaussians(means, gt_pts, gt_labels, dist_cap=GT_DIST_CAP,
                    chunk=1024, gt_chunk=200_000):
    """Nearest-GT label per Gaussian, chunked over BOTH sides so peak
    memory stays ~chunk*gt_chunk floats (the unchunked version tried to
    allocate 15 GB on a 2M-point cloud)."""
    N = means.shape[0]
    labels = torch.full((N,), -1, dtype=torch.long, device=means.device)
    for s in range(0, N, chunk):
        m = means[s:s + chunk]
        best_d = torch.full((m.shape[0],), float("inf"), device=means.device)
        best_l = torch.full((m.shape[0],), -1, dtype=torch.long,
                            device=means.device)
        for g in range(0, gt_pts.shape[0], gt_chunk):
            d = torch.cdist(m, gt_pts[g:g + gt_chunk])
            mind, argd = d.min(dim=1)
            better = mind < best_d
            best_d[better] = mind[better]
            best_l[better] = gt_labels[g:g + gt_chunk][argd[better]]
        best_l[best_d > dist_cap] = -1
        labels[s:s + chunk] = best_l
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--label-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=os.path.join(
        PROJECT_ROOT, "results", "sweep", "miou_slam_checkpoint.pt"))
    ap.add_argument("--from-checkpoint", action="store_true",
                    help="skip SLAM; evaluate a previously saved map")
    args = ap.parse_args()

    import rerun_d16_all as R
    from dynlang_slam.core.gaussians import GaussianMap
    from dynlang_slam.slam.pipeline import SLAMPipeline
    from dynlang_slam.data.replica import get_replica_intrinsics

    R._seed_everything(args.seed)
    device = "cuda"

    print(f"Loading {args.frames} vMAP room_0 frames (stride {args.stride})...")
    frames = load_frames(args.frames, args.stride)
    names = class_names()
    intr = get_replica_intrinsics()   # fx=fy=600, cx=599.5, cy=339.5, 1200x680
    K = intr["K"].to(device)

    # --- SLAM (semantic images never enter the pipeline) ---
    cfg = R.build_cfg(R.SCENES[2], "full")
    gmap = GaussianMap(
        sh_degree=cfg.gaussians.sh_degree,
        lang_feat_dim=cfg.gaussians.lang_feat_dim,
        init_opacity=cfg.gaussians.init_opacity,
        device=device,
    )
    slam = SLAMPipeline(cfg=cfg, intrinsics=intr, device=device)

    if args.from_checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False)
        gmap.load_state_dict_compact(ckpt["map"])
        slam._init_language_pipeline()
        slam._autoencoder.load_state_dict(ckpt["ae"])
        slam._autoencoder.freeze()
        ate = ckpt["ate_rmse_cm"]
        print(f"Loaded checkpoint: ATE={ate:.2f}cm G={gmap.num_gaussians}")
    else:
        slam.process_first_frame(gmap, frames[0])
        t0 = time.time()
        for i in range(1, len(frames)):
            slam.process_frame(gmap, frames[i], use_gt_pose=False)
            if i % 25 == 0:
                print(f"  frame {i}/{len(frames)-1}", flush=True)
        gt_poses = [f["pose"].to(device) for f in frames]
        ate = slam.compute_ate_rmse(gt_poses) * 100
        print(f"SLAM done: ATE={ate:.2f}cm G={gmap.num_gaussians} "
              f"t={time.time()-t0:.0f}s")
        torch.save({
            "map": gmap.state_dict_compact(),
            "ae": slam._autoencoder.state_dict(),
            "ate_rmse_cm": float(ate),
            "frames": args.frames, "stride": args.stride, "seed": args.seed,
        }, args.checkpoint)
        print(f"checkpoint saved: {args.checkpoint}")

    # --- GT labeling ---
    gt_pts, gt_lab = build_gt_pointcloud(frames, K, args.label_every, device)
    print(f"GT point cloud: {gt_pts.shape[0]} points")
    g_labels = label_gaussians(gmap.means.data, gt_pts, gt_lab)

    present = [int(c) for c in torch.unique(gt_lab).tolist()
               if (gt_lab == c).sum() >= MIN_GT_POINTS and c in names]
    print(f"classes present (>= {MIN_GT_POINTS} GT pts): "
          f"{[names[c] for c in present]}")

    # --- open-vocabulary classification ---
    norm_ok = gmap.lang_feats.data.norm(dim=-1) > 0.1
    score_rows = []
    for c in present:
        r = slam.query_3d(gmap, names[c], top_k=1)
        score_rows.append(r["scores"])
    scores = torch.stack(score_rows)                     # (C, N)
    pred_idx = scores.argmax(dim=0)                      # (N,)
    pred = torch.tensor(present, device=device)[pred_idx]

    evalmask = (g_labels > 0) & norm_ok
    print(f"evaluated Gaussians: {int(evalmask.sum())} of {gmap.num_gaussians} "
          f"(GT-labeled and norm-gated)")

    print(f"\n{'class':>16} {'IoU':>7} {'GT n':>7}")
    ious = []
    for c in present:
        gt_c = (g_labels == c) & evalmask
        pd_c = (pred == c) & evalmask
        inter = (gt_c & pd_c).sum().item()
        union = (gt_c | pd_c).sum().item()
        if union == 0:
            continue
        iou = inter / union
        ious.append(iou)
        print(f"{names[c]:>16} {iou:>7.3f} {int(gt_c.sum()):>7}")
    miou = float(np.mean(ious)) if ious else 0.0
    print(f"\nmIoU (zero-shot, {len(ious)} classes): {miou:.3f}")

    out = os.path.join(PROJECT_ROOT, "results", "sweep", "exp_miou.jsonl")
    with open(out, "a") as f:
        f.write(json.dumps({
            "experiment": "miou_room0_vmap", "seed": args.seed,
            "frames": args.frames, "stride": args.stride,
            "ate_rmse_cm": float(ate), "gaussians": int(gmap.num_gaussians),
            "n_classes": len(ious), "miou": miou,
            "per_class": {names[c]: float(i) for c, i in zip(present, ious)},
        }) + "\n")
    print("RESULT written to", out)


if __name__ == "__main__":
    main()
