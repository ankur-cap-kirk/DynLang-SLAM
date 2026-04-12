"""Camera tracking module for DynLang-SLAM.

Optimizes the 6-DOF camera pose by minimizing the photometric and geometric
loss between the rendered Gaussian map and the observed RGB-D frame.
"""

import torch
import torch.nn as nn

from ..core.gaussians import GaussianMap
from ..core.renderer import GaussianRenderer
from ..core.losses import compute_losses
from ..utils.camera import pose_to_matrix


class Tracker:
    """Estimates camera pose by optimizing against the current Gaussian map."""

    def __init__(
        self,
        renderer: GaussianRenderer,
        num_iterations: int = 40,
        lr_pose: float = 0.01,
        loss_weights: dict = None,
        device: str = "cuda",
        render_downscale: int = 4,
        lr_end_factor: float = 0.05,
        coarse_to_fine: bool = True,
        coarse_downscale: int = 4,
        coarse_ratio: float = 0.6,
    ):
        self.renderer = renderer
        self.num_iterations = num_iterations
        self.lr_pose = lr_pose
        self.loss_weights = loss_weights or {
            "rgb_weight": 0.5,
            "depth_weight": 1.0,
            "ssim_weight": 0.2,
        }
        self.device = device
        self.render_downscale = render_downscale
        self.lr_end_factor = lr_end_factor  # LR decays to lr_pose * lr_end_factor
        # Coarse-to-fine: first coarse_ratio of iterations at coarse_downscale,
        # then remaining at render_downscale (GS-SLAM inspired)
        self.coarse_to_fine = coarse_to_fine
        self.coarse_downscale = coarse_downscale
        self.coarse_ratio = coarse_ratio

    def track(
        self,
        gaussian_map: GaussianMap,
        gt_rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        init_pose: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """Estimate camera pose for a new frame.

        Args:
            gaussian_map: Current Gaussian map (frozen during tracking)
            gt_rgb: (3, H, W) observed RGB image [0, 1]
            gt_depth: (1, H, W) observed depth in meters
            K: (3, 3) camera intrinsic matrix
            width, height: image dimensions
            init_pose: (4, 4) initial camera-to-world pose estimate
            mask: optional (H, W) binary mask (1 = static, 0 = dynamic)

        Returns:
            estimated_pose: (4, 4) optimized camera-to-world pose
            info: dict with tracking metrics
        """
        # Precompute downsampled GT images for both coarse and fine stages
        def _downsample_gt(ds):
            if ds > 1:
                rgb_ds = torch.nn.functional.interpolate(
                    gt_rgb.unsqueeze(0), scale_factor=1.0/ds,
                    mode='bilinear', align_corners=False).squeeze(0)
                depth_ds = torch.nn.functional.interpolate(
                    gt_depth.unsqueeze(0), scale_factor=1.0/ds,
                    mode='nearest').squeeze(0)
            else:
                rgb_ds = gt_rgb
                depth_ds = gt_depth
            return rgb_ds.permute(1, 2, 0), depth_ds.squeeze(0)

        # Coarse-to-fine: determine iteration split
        if self.coarse_to_fine and self.coarse_downscale > self.render_downscale:
            coarse_iters = int(self.num_iterations * self.coarse_ratio)
            fine_iters = self.num_iterations - coarse_iters
            stages = [
                (coarse_iters, self.coarse_downscale),
                (fine_iters, self.render_downscale),
            ]
        else:
            stages = [(self.num_iterations, self.render_downscale)]

        # Precompute GT for each unique downscale
        gt_cache = {}
        for _, ds in stages:
            if ds not in gt_cache:
                gt_cache[ds] = _downsample_gt(ds)

        # Parametrize pose as quaternion (4) + translation (3) for optimization
        R_init = init_pose[:3, :3]
        t_init = init_pose[:3, 3].clone()
        quat_init = _matrix_to_quaternion(R_init)

        # Learnable pose parameters
        opt_quat = quat_init.clone().detach().requires_grad_(True)
        opt_trans = t_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([opt_quat, opt_trans], lr=self.lr_pose)

        best_loss = float("inf")
        best_quat = opt_quat.data.clone()
        best_trans = opt_trans.data.clone()

        # Freeze Gaussian parameters during tracking
        for param in gaussian_map.parameters():
            param.requires_grad_(False)

        # Dynamic loss weighting: depth-heavy early (convex basin),
        # RGB/SSIM-heavy late (photometric refinement)
        base_rgb_w = self.loss_weights.get("rgb_weight", 0.5)
        base_depth_w = self.loss_weights.get("depth_weight", 1.0)
        base_ssim_w = self.loss_weights.get("ssim_weight", 0.2)
        total_iters = self.num_iterations

        global_iter = 0
        for stage_iters, ds in stages:
            gt_rgb_hw, gt_depth_hw = gt_cache[ds]
            for _ in range(stage_iters):
                optimizer.zero_grad()

                # Interpolate loss weights: t goes from 0.0 (start) to 1.0 (end)
                t = global_iter / max(total_iters - 1, 1)
                # Literature-based: constant depth, rising RGB/SSIM
                # Matches SplaTAM pattern (depth=1.0, rgb=0.5) early,
                # then gradually adds photometric refinement
                depth_scale = 1.0              # constant, never reduce depth anchor
                rgb_scale = 0.5 + 0.5 * t     # 0.5 -> 1.0
                ssim_scale = 0.5 + 0.5 * t    # 0.5 -> 1.0
                iter_weights = {
                    "rgb_weight": base_rgb_w * rgb_scale,
                    "depth_weight": base_depth_w * depth_scale,
                    "ssim_weight": base_ssim_w * ssim_scale,
                }

                # Build pose matrix from optimizable parameters
                pose_c2w = pose_to_matrix(opt_quat, opt_trans)
                viewmat = torch.inverse(pose_c2w)

                # Render at current stage's resolution
                rendered = self.renderer(
                    gaussian_map, viewmat, K, width, height, render_lang=False,
                    downscale=ds,
                )

                # Compute loss with dynamic weights
                loss, loss_dict = compute_losses(
                    rendered, gt_rgb_hw, gt_depth_hw, iter_weights, mask
                )

                loss.backward()
                optimizer.step()

                # Normalize quaternion to prevent magnitude drift
                with torch.no_grad():
                    opt_quat.data = torch.nn.functional.normalize(opt_quat.data, dim=-1)

                if loss_dict["total"] < best_loss:
                    best_loss = loss_dict["total"]
                    best_quat = opt_quat.data.clone()
                    best_trans = opt_trans.data.clone()

                global_iter += 1

        # Re-enable gradients for Gaussians
        for param in gaussian_map.parameters():
            param.requires_grad_(True)

        # Build final pose
        final_pose = pose_to_matrix(best_quat, best_trans).detach()

        info = {
            "final_loss": best_loss,
            "iterations": self.num_iterations,
        }
        return final_pose, info


def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z].

    Robust implementation handling all edge cases.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0 + 1e-8)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2] + 1e-8)
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2] + 1e-8)
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1] + 1e-8)
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return torch.stack([w, x, y, z])
