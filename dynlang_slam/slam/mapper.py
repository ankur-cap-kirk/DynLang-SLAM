"""Mapping module for DynLang-SLAM.

Optimizes Gaussian parameters and adds new Gaussians for unobserved regions
using a silhouette-based expansion mechanism (following SplaTAM).
"""

import torch

from ..core.gaussians import GaussianMap
from ..core.renderer import GaussianRenderer
from ..core.losses import compute_losses, language_loss
from ..utils.camera import depth_to_points, fast_se3_inverse, pose_to_matrix, matrix_to_pose


class Mapper:
    """Optimizes the Gaussian map given a set of keyframes."""

    def __init__(
        self,
        renderer: GaussianRenderer,
        num_iterations: int = 60,
        lr_means: float = 0.0001,
        lr_scales: float = 0.005,
        lr_quats: float = 0.001,
        lr_opacities: float = 0.05,
        lr_colors: float = 0.0025,
        lr_lang_feats: float = 0.001,
        loss_weights: dict = None,
        new_gaussian_thresh: float = 0.5,
        device: str = "cuda",
        render_downscale: int = 4,
        lr_end_factor: float = 0.1,
        densify_grad_thresh: float = 0.0002,
        densify_interval: int = 5,
        densify_max_scale: float = 0.05,
        iso_reg_weight: float = 0.05,
        use_soft_dynamic: bool = False,
        refine_poses: bool = False,
        lr_pose_trans: float = 1.0e-4,
        lr_pose_quat: float = 5.0e-4,
        pose_prior_weight: float = 10.0,
        use_hard_rgb_mask: bool = True,
        reliability_thresh: float = 0.5,
    ):
        self.renderer = renderer
        self.iso_reg_weight = iso_reg_weight
        self.num_iterations = num_iterations
        self.lr_config = {
            "means": lr_means,
            "scales": lr_scales,
            "quats": lr_quats,
            "opacities": lr_opacities,
            "colors": lr_colors,
            "lang_feats": lr_lang_feats,
        }
        # Local BA (intervention A1): refine non-anchor keyframe poses
        # jointly with Gaussians. Oldest KF in window stays fixed as
        # gauge anchor to prevent trajectory rigid-body drift.
        self.refine_poses = refine_poses
        self.lr_pose_trans = lr_pose_trans
        self.lr_pose_quat = lr_pose_quat
        # MonoGS-style pose prior: penalize refinement away from the
        # tracker's initial estimate. Without this, under-constrained
        # regions (few valid pixels, dynamic masking, textureless walls)
        # let the pose slide into local minima that minimize photometric
        # loss on remaining pixels but aren't geometrically correct.
        # Weight interpretation: each meter of translation drift costs
        # `pose_prior_weight` loss units (vs photometric losses ~1).
        self.pose_prior_weight = pose_prior_weight
        # D1 knobs: hard alpha gate for RGB vs legacy soft weighting
        self.use_hard_rgb_mask = use_hard_rgb_mask
        self.reliability_thresh = reliability_thresh
        self.loss_weights = loss_weights or {
            "rgb_weight": 0.5,
            "depth_weight": 1.0,
            "ssim_weight": 0.2,
        }
        self.new_gaussian_thresh = new_gaussian_thresh
        self.device = device
        self.render_downscale = render_downscale
        self.lr_end_factor = lr_end_factor  # LR decays to lr * lr_end_factor
        self.densify_grad_thresh = densify_grad_thresh
        self.densify_interval = densify_interval
        self.densify_max_scale = densify_max_scale
        self.use_soft_dynamic = use_soft_dynamic
        self._map_call_count = 0

    def map(
        self,
        gaussian_map: GaussianMap,
        frames: list[dict],
        poses: list[torch.Tensor],
        K: torch.Tensor,
        width: int,
        height: int,
        add_new: bool = True,
        lang_feature_maps: list[dict] = None,
        lang_weight: float = 0.0,
        masks: list[torch.Tensor] = None,
    ) -> dict:
        """Optimize Gaussian map parameters using keyframe observations.

        Args:
            gaussian_map: The Gaussian map to optimize
            frames: list of dicts with 'rgb' (3,H,W) and 'depth' (1,H,W)
            poses: list of (4,4) camera-to-world poses
            K: (3,3) camera intrinsic matrix
            width, height: image dimensions
            add_new: whether to add new Gaussians for unobserved regions

        Returns:
            info dict with mapping metrics
        """
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()

        # Add new Gaussians for unobserved regions (before optimization)
        total_added = 0
        if add_new:
            for fi, (frame, pose) in enumerate(zip(frames, poses)):
                frame_mask = masks[fi] if masks is not None and fi < len(masks) else None
                n_added = self._expand_map(
                    gaussian_map, frame, pose, K, width, height, fx, fy, cx, cy,
                    mask=frame_mask,
                )
                total_added += n_added

        # Check if we should optimize language features
        use_lang = (lang_weight > 0 and lang_feature_maps is not None
                     and len(lang_feature_maps) > 0)

        # Set up optimizer for Gaussian parameters
        param_groups = [
            {"params": [gaussian_map.means], "lr": self.lr_config["means"]},
            {"params": [gaussian_map.scales], "lr": self.lr_config["scales"]},
            {"params": [gaussian_map.quats], "lr": self.lr_config["quats"]},
            {"params": [gaussian_map.opacities], "lr": self.lr_config["opacities"]},
            {"params": [gaussian_map.colors], "lr": self.lr_config["colors"]},
        ]
        if use_lang:
            param_groups.append(
                {"params": [gaussian_map.lang_feats], "lr": self.lr_config["lang_feats"]}
            )

        # Local BA: promote non-anchor keyframe poses to learnable params.
        # poses[0] is the gauge anchor (fixed). poses[1:] become (quat, trans)
        # pairs jointly optimized with the Gaussian parameters.
        do_refine = self.refine_poses and len(poses) > 1
        pose_params = []  # list of (quat_param, trans_param) for poses[1:]
        pose_init_quats = []  # frozen init quats (for pose prior loss)
        pose_init_trans = []  # frozen init translations (for pose prior loss)
        if do_refine:
            for p in poses[1:]:
                q_init, t_init = matrix_to_pose(p.detach())
                q_param = q_init.clone().detach().requires_grad_(True)
                t_param = t_init.clone().detach().requires_grad_(True)
                pose_params.append((q_param, t_param))
                pose_init_quats.append(q_init.detach().clone())
                pose_init_trans.append(t_init.detach().clone())
            quat_list = [qp for qp, _ in pose_params]
            trans_list = [tp for _, tp in pose_params]
            param_groups.append({"params": quat_list, "lr": self.lr_pose_quat})
            param_groups.append({"params": trans_list, "lr": self.lr_pose_trans})

        lr_keys = ["means", "scales", "quats", "opacities", "colors"]
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_iterations,
            eta_min=min(self.lr_config[k] * self.lr_end_factor
                        for k in lr_keys),
        )

        total_loss = 0.0
        loss_info = {}

        for iteration in range(self.num_iterations):
            optimizer.zero_grad()

            # Rebuild pose tensors from learnable params (poses[0] is anchor)
            if do_refine:
                active_poses = [poses[0]]
                for q_param, t_param in pose_params:
                    active_poses.append(pose_to_matrix(q_param, t_param))
            else:
                active_poses = poses

            iter_loss = 0.0
            loss_lang_total = 0.0
            for fi, (frame, pose) in enumerate(zip(frames, active_poses)):
                ds = self.render_downscale
                if ds > 1:
                    gt_rgb_ds = torch.nn.functional.interpolate(
                        frame["rgb"].unsqueeze(0), scale_factor=1.0/ds,
                        mode='bilinear', align_corners=False).squeeze(0)
                    gt_depth_ds = torch.nn.functional.interpolate(
                        frame["depth"].unsqueeze(0), scale_factor=1.0/ds,
                        mode='nearest').squeeze(0)
                else:
                    gt_rgb_ds = frame["rgb"]
                    gt_depth_ds = frame["depth"]
                gt_rgb = gt_rgb_ds.permute(1, 2, 0)
                gt_depth = gt_depth_ds.squeeze(0)

                # fast_se3_inverse uses in-place ops which break the autograd
                # graph through learnable pose params. Use torch.inverse
                # (differentiable, negligible cost on 4x4) when refining.
                if do_refine and fi > 0:
                    viewmat = torch.inverse(pose)
                else:
                    viewmat = fast_se3_inverse(pose)

                rendered = self.renderer(
                    gaussian_map, viewmat, K, width, height,
                    render_lang=use_lang, downscale=ds,
                )

                # Dynamic mask: downscale to match render resolution
                frame_mask = None
                if masks is not None and fi < len(masks) and masks[fi] is not None:
                    if ds > 1:
                        frame_mask = torch.nn.functional.interpolate(
                            masks[fi].unsqueeze(0).unsqueeze(0),
                            scale_factor=1.0/ds, mode='nearest',
                        ).squeeze(0).squeeze(0)
                    else:
                        frame_mask = masks[fi]

                loss, loss_dict = compute_losses(
                    rendered, gt_rgb, gt_depth, self.loss_weights,
                    mask=frame_mask,
                    use_soft_dynamic=self.use_soft_dynamic,
                    use_hard_rgb_mask=self.use_hard_rgb_mask,
                    reliability_thresh=self.reliability_thresh,
                )
                iter_loss = iter_loss + loss

                # Language loss: supervise rendered lang_feats against target
                if use_lang and fi < len(lang_feature_maps) and "lang_feats" in rendered:
                    gt_lang = lang_feature_maps[fi]  # (H, W, D) on device
                    # Downscale target to match render resolution
                    if ds > 1:
                        # (H, W, D) -> (1, D, H, W) -> interpolate -> (H', W', D)
                        gt_lang_ds = torch.nn.functional.interpolate(
                            gt_lang.permute(2, 0, 1).unsqueeze(0),
                            scale_factor=1.0/ds,
                            mode='bilinear', align_corners=False,
                        ).squeeze(0).permute(1, 2, 0)
                    else:
                        gt_lang_ds = gt_lang
                    loss_l = language_loss(
                        rendered["lang_feats"], gt_lang_ds, rendered["alpha"]
                    )
                    iter_loss = iter_loss + lang_weight * loss_l
                    loss_lang_total += loss_l.item()

            # Isotropic regularization: penalize elongated Gaussians (MonoGS)
            activated_scales = torch.exp(gaussian_map.scales)  # (N, 3)
            scale_mean = activated_scales.mean(dim=-1, keepdim=True)  # (N, 1)
            loss_iso = torch.abs(activated_scales - scale_mean).mean()

            # Average over frames + regularization
            iso_w = getattr(self, 'iso_reg_weight', 0.05)
            iter_loss = iter_loss / len(frames) + iso_w * loss_iso

            # Pose prior (MonoGS): penalize refinement away from tracker's
            # initial estimate. Keeps under-constrained poses (few valid
            # pixels, dynamic masking, textureless regions) from sliding
            # into bad local minima. Quadratic so small corrections are
            # nearly free but large drifts are heavily penalized.
            if do_refine and self.pose_prior_weight > 0:
                pose_prior_loss = 0.0
                for (q_param, t_param), q_init, t_init in zip(
                    pose_params, pose_init_quats, pose_init_trans,
                ):
                    pose_prior_loss = pose_prior_loss + \
                        ((t_param - t_init) ** 2).sum() + \
                        ((q_param - q_init) ** 2).sum()
                iter_loss = iter_loss + self.pose_prior_weight * pose_prior_loss

            iter_loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss = iter_loss.item()
            loss_info = loss_dict

        # Build final refined-poses list (anchor + param-derived) for writeback
        # and for downstream contamination / densification calls.
        if do_refine:
            with torch.no_grad():
                final_poses = [poses[0]]
                for q_param, t_param in pose_params:
                    final_poses.append(pose_to_matrix(q_param, t_param).detach())
        else:
            final_poses = poses

        # Mark Gaussians contaminated by dynamic objects
        if masks is not None:
            with torch.no_grad():
                for fi, (frame, pose) in enumerate(zip(frames, final_poses)):
                    if fi >= len(masks) or masks[fi] is None:
                        continue
                    if (masks[fi] < 0.5).sum() == 0:
                        continue  # no dynamic pixels in this frame
                    viewmat = fast_se3_inverse(pose)
                    gaussian_map.mark_contaminated(
                        viewmat, K, width, height, masks[fi],
                    )

        # Densify based on position gradients + rendering error (HF-SLAM style)
        self._map_call_count += 1
        densify_info = {"split": 0, "cloned": 0}
        if self._map_call_count % self.densify_interval == 0:
            if gaussian_map.means.grad is not None:
                grad_norms = gaussian_map.means.grad.detach().norm(dim=-1)

                # Rendering-guided boost: also densify Gaussians contributing to
                # high-error regions (even if gradient is below threshold)
                # Use the last frame's rendering error as a signal
                with torch.no_grad():
                    frame, pose = frames[-1], final_poses[-1]
                    ds = max(self.render_downscale, 4)  # at least 4x for speed
                    gt_rgb_ds = torch.nn.functional.interpolate(
                        frame["rgb"].unsqueeze(0), scale_factor=1.0/ds,
                        mode='bilinear', align_corners=False).squeeze(0)
                    gt_depth_ds = torch.nn.functional.interpolate(
                        frame["depth"].unsqueeze(0), scale_factor=1.0/ds,
                        mode='nearest').squeeze(0)
                    viewmat = fast_se3_inverse(pose)
                    rendered = self.renderer(
                        gaussian_map, viewmat, K, width, height,
                        render_lang=False, downscale=ds,
                    )
                    # Per-pixel color error
                    color_err = torch.abs(
                        rendered["rgb"] - gt_rgb_ds.permute(1, 2, 0)
                    ).mean(dim=-1)  # (H, W)
                    alpha_map = rendered["alpha"].squeeze(-1)
                    # High-error or low-coverage pixels need densification
                    need_densify = (color_err > 0.3) | (alpha_map < 0.5)
                    error_ratio = need_densify.float().mean().item()

                # If high error ratio, lower the gradient threshold temporarily
                effective_thresh = self.densify_grad_thresh
                if error_ratio > 0.1:
                    effective_thresh = self.densify_grad_thresh * 0.5

                densify_info = gaussian_map.densify(
                    grad_norms, effective_thresh, self.densify_max_scale
                )

        # Prune low-opacity Gaussians periodically
        n_pruned = self._prune_gaussians(gaussian_map, min_opacity=0.005)

        info = {
            "final_loss": total_loss,
            "gaussians_added": total_added,
            "gaussians_pruned": n_pruned,
            "densified_split": densify_info["split"],
            "densified_cloned": densify_info["cloned"],
            "total_gaussians": gaussian_map.num_gaussians,
            "iterations": self.num_iterations,
            "lang_loss": loss_lang_total / max(self.num_iterations, 1) if use_lang else 0.0,
        }
        info.update({f"loss_{k}": v for k, v in loss_info.items()})

        # Local BA: return refined poses for pipeline writeback, plus a
        # diagnostic on how far each non-anchor pose moved from its
        # tracker-supplied initialization.
        if do_refine:
            info["refined_poses"] = final_poses
            with torch.no_grad():
                trans_deltas = []
                for orig, refined in zip(poses[1:], final_poses[1:]):
                    d = (refined[:3, 3] - orig[:3, 3]).norm().item()
                    trans_deltas.append(d)
                info["pose_refine_trans_max_m"] = max(trans_deltas) if trans_deltas else 0.0
                info["pose_refine_trans_mean_m"] = (
                    sum(trans_deltas) / len(trans_deltas) if trans_deltas else 0.0
                )
        return info

    def _expand_map(
        self,
        gaussian_map: GaussianMap,
        frame: dict,
        pose: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        fx: float, fy: float, cx: float, cy: float,
        mask: torch.Tensor = None,
    ) -> int:
        """Add new Gaussians for regions not covered by existing map.

        Uses silhouette rendering: pixels with low alpha need new Gaussians.
        """
        with torch.no_grad():
            viewmat = torch.inverse(pose)
            alpha = self.renderer.render_silhouette(
                gaussian_map, viewmat, K, width, height
            )

            # Find unmapped pixels (low alpha AND valid depth)
            depth = frame["depth"].squeeze(0)  # (H, W)
            rgb = frame["rgb"].permute(1, 2, 0)  # (H, W, 3)

            unmapped = (alpha < self.new_gaussian_thresh) & (depth > 0)

            # Exclude dynamic pixels from expansion
            if mask is not None:
                unmapped = unmapped & (mask > 0.5)

            if unmapped.sum() == 0:
                return 0

            # Downsample unmapped regions to avoid adding too many Gaussians
            downsample = 4
            unmapped_ds = unmapped[::downsample, ::downsample]
            depth_ds = depth[::downsample, ::downsample]
            rgb_ds = rgb[::downsample, ::downsample]

            if unmapped_ds.sum() == 0:
                return 0

            # Unproject unmapped pixels to 3D
            H_ds, W_ds = depth_ds.shape
            v, u = torch.meshgrid(
                torch.arange(H_ds, device=depth.device, dtype=torch.float32),
                torch.arange(W_ds, device=depth.device, dtype=torch.float32),
                indexing="ij",
            )

            fx_ds = fx / downsample
            fy_ds = fy / downsample
            cx_ds = cx / downsample
            cy_ds = cy / downsample

            z = depth_ds[unmapped_ds]
            u_valid = u[unmapped_ds]
            v_valid = v[unmapped_ds]

            x = (u_valid - cx_ds) * z / fx_ds
            y = (v_valid - cy_ds) * z / fy_ds

            points_cam = torch.stack([x, y, z], dim=-1)  # (M, 3)

            # Transform to world space
            R = pose[:3, :3]
            t = pose[:3, 3]
            new_means = points_cam @ R.T + t.unsqueeze(0)
            new_colors = rgb_ds[unmapped_ds]  # (M, 3)

            return gaussian_map.add_gaussians(new_means, new_colors)

    def _prune_gaussians(self, gaussian_map: GaussianMap, min_opacity: float = 0.005) -> int:
        """Remove Gaussians with very low opacity."""
        if gaussian_map.num_gaussians == 0:
            return 0
        with torch.no_grad():
            opacities = torch.sigmoid(gaussian_map.opacities.data).squeeze(-1)
            prune_mask = opacities < min_opacity
            if prune_mask.sum() > 0:
                return gaussian_map.prune(prune_mask)
        return 0
