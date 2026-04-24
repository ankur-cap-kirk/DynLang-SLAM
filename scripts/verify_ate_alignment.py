"""Verification of Umeyama-aligned ATE vs raw ATE.

Three sanity checks:

  1. Unit test: if est = R_gt @ gt + t_gt + noise, the aligned ATE should
     equal the noise RMS and the raw ATE should be much larger.

  2. Unit test: if est == gt, both aligned and raw must be exactly 0.

  3. Re-run a short BONN person_tracking / Replica room0 clip end-to-end
     and print both raw and aligned ATE so the drop can be inspected.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

from dynlang_slam.slam.pipeline import SLAMPipeline


# ------------------------------- unit tests --------------------------------


def _random_rot(rng: np.random.Generator) -> np.ndarray:
    # Random rotation via axis-angle (small angle, any axis).
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(0.1, 1.2)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def test_identity():
    N = 50
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((N, 3))
    R, t = SLAMPipeline._umeyama_se3(pts, pts)
    assert np.allclose(R, np.eye(3), atol=1e-10), "R should be I for identical trajectories"
    assert np.allclose(t, 0.0, atol=1e-10), "t should be 0 for identical trajectories"
    print("PASS  test_identity: R=I, t=0 when est == gt")


def test_rigid_recovery():
    rng = np.random.default_rng(42)
    N = 200
    gt = rng.standard_normal((N, 3))
    R_true = _random_rot(rng)
    t_true = rng.standard_normal(3) * 2.0
    est = (R_true @ gt.T).T + t_true  # est is a rigid copy of gt
    # Recover inverse: est -> gt should be (R_true^T, -R_true^T @ t_true)
    R, t = SLAMPipeline._umeyama_se3(est, gt)
    recon = (R @ est.T).T + t
    err = np.linalg.norm(recon - gt, axis=1).max()
    assert err < 1e-8, f"Alignment should be exact for noise-free rigid data, got {err:.2e}"
    print(f"PASS  test_rigid_recovery: max residual after alignment = {err:.2e}")


def test_noise_rms():
    rng = np.random.default_rng(7)
    N = 300
    gt = rng.standard_normal((N, 3))
    R_true = _random_rot(rng)
    t_true = np.array([5.0, -3.0, 1.0])
    noise = rng.standard_normal((N, 3)) * 0.05  # 5 cm noise
    est = (R_true @ gt.T).T + t_true + noise
    # Raw ATE: dominated by t_true magnitude
    raw = float(np.sqrt(np.mean(np.linalg.norm(est - gt, axis=1) ** 2)))
    # Aligned ATE: should equal noise RMS (~0.05 * sqrt(3) ~= 0.087 m per-frame)
    R, t = SLAMPipeline._umeyama_se3(est, gt)
    recon = (R @ est.T).T + t
    aligned = float(np.sqrt(np.mean(np.linalg.norm(recon - gt, axis=1) ** 2)))
    expected_noise_rms = float(np.sqrt(np.mean(np.linalg.norm(noise, axis=1) ** 2)))
    print(f"PASS  test_noise_rms: raw={raw:.3f} m, aligned={aligned:.3f} m, "
          f"expected noise RMS={expected_noise_rms:.3f} m "
          f"(aligned should match expected, raw should be much larger)")
    assert raw > 1.0, "raw ATE should be dominated by the 5 m translation offset"
    assert abs(aligned - expected_noise_rms) < 1e-3, \
        f"aligned ATE should equal noise RMS, got {aligned:.4f} vs {expected_noise_rms:.4f}"


# ------------------------- end-to-end comparison ---------------------------


def run_bonn_compare(n_frames: int = 100, stride: int = 2):
    """Re-run a short BONN person_tracking clip and print raw vs aligned ATE."""
    from dynlang_slam.utils.config import load_config
    from dynlang_slam.data.tum import TUMDataset, get_bonn_intrinsics
    from dynlang_slam.core.gaussians import GaussianMap

    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"), [])
    cfg.dataset.type = "tum"
    cfg.dataset.image_height = 480
    cfg.dataset.image_width = 640
    cfg.dataset.depth_scale = 5000.0
    cfg.dynamic.enabled = True
    cfg.dynamic.yolo_model = "yolov8n-seg"
    cfg.language.enabled = False

    dataset = TUMDataset(
        data_dir=os.path.join(PROJECT_ROOT, "data", "BONN", "rgbd_bonn_person_tracking"),
        height=480, width=640, depth_scale=5000.0,
        max_frames=n_frames, stride=stride,
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
    gt_poses = [dataset[i]["pose"].to(device) for i in range(len(dataset))]

    raw_cm     = slam.compute_ate_rmse(gt_poses, align=False) * 100
    aligned_cm = slam.compute_ate_rmse(gt_poses, align=True)  * 100
    dt = time.time() - t0

    print("-" * 60)
    print(f"BONN person_tracking  n_frames={len(dataset)}  runtime={dt:.1f}s")
    print(f"  raw      ATE-RMSE: {raw_cm:.2f} cm   (old reporting)")
    print(f"  ALIGNED  ATE-RMSE: {aligned_cm:.2f} cm   (SLAM-standard, TUM/Sturm 2012)")
    print("-" * 60)
    return raw_cm, aligned_cm


if __name__ == "__main__":
    print("== Unit tests for _umeyama_se3 ==")
    test_identity()
    test_rigid_recovery()
    test_noise_rms()
    print()

    if os.environ.get("DYNLANG_SKIP_BONN", "0") != "1":
        print("== End-to-end: re-run BONN clip with new reporting ==")
        run_bonn_compare(
            n_frames=int(os.environ.get("DYNLANG_N_FRAMES", "100")),
            stride=int(os.environ.get("DYNLANG_STRIDE", "2")),
        )
