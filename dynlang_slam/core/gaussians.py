"""3D Gaussian representation for DynLang-SLAM.

Each Gaussian stores:
    - means (3,): 3D position
    - scales (3,): log-space scale
    - quats (4,): rotation quaternion [w, x, y, z]
    - opacities (1,): sigmoid-space opacity
    - colors (C,): RGB or SH coefficients
    - lang_feats (D,): compressed CLIP language features
"""

import torch
import torch.nn as nn
import numpy as np

from ..utils.camera import depth_to_points


class GaussianMap(nn.Module):
    """Manages a collection of 3D Gaussians as the scene map."""

    def __init__(
        self,
        sh_degree: int = 0,
        lang_feat_dim: int = 16,
        init_opacity: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.lang_feat_dim = lang_feat_dim
        self.init_opacity = init_opacity
        self.device = device

        # Number of SH coefficients per color channel
        self.n_sh_coeffs = (sh_degree + 1) ** 2
        self.color_dim = self.n_sh_coeffs * 3  # RGB channels

        # Initialize empty parameter tensors
        self._means = nn.ParameterList()
        self._scales = nn.ParameterList()
        self._quats = nn.ParameterList()
        self._opacities = nn.ParameterList()
        self._colors = nn.ParameterList()
        self._lang_feats = nn.ParameterList()

        # Flat storage (rebuilt when needed)
        self.means = torch.empty(0, 3, device=device)
        self.scales = torch.empty(0, 3, device=device)
        self.quats = torch.empty(0, 4, device=device)
        self.opacities = torch.empty(0, 1, device=device)
        self.colors = torch.empty(0, self.color_dim, device=device)
        self.lang_feats = torch.empty(0, lang_feat_dim, device=device)

        self._num_gaussians = 0

        # Contamination tracking for dynamic objects (no gradients)
        self.contamination_count = torch.zeros(0, dtype=torch.long, device=device)

        # Bayesian dynamic belief: per-Gaussian probability of being dynamic (BDGS-SLAM)
        self.dynamic_belief = torch.zeros(0, dtype=torch.float32, device=device)

    @property
    def num_gaussians(self) -> int:
        return self._num_gaussians

    def initialize_from_depth(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor,
        pose: torch.Tensor,
        fx: float, fy: float, cx: float, cy: float,
        downsample: int = 4,
    ) -> None:
        """Initialize Gaussians from the first RGB-D frame.

        Args:
            depth: (1, H, W) or (H, W) depth map in meters
            rgb: (3, H, W) RGB image, range [0, 1]
            pose: (4, 4) camera-to-world transform
            fx, fy, cx, cy: camera intrinsics
            downsample: spatial downsampling factor for initial points
        """
        if depth.dim() == 3:
            depth = depth.squeeze(0)
        if rgb.dim() == 3:
            rgb_hw = rgb.permute(1, 2, 0)  # (H, W, 3)
        else:
            rgb_hw = rgb

        H, W = depth.shape

        # Downsample for initial point cloud
        depth_ds = depth[::downsample, ::downsample]
        rgb_ds = rgb_hw[::downsample, ::downsample]

        # Adjusted intrinsics for downsampled grid
        fx_ds = fx / downsample
        fy_ds = fy / downsample
        cx_ds = cx / downsample
        cy_ds = cy / downsample

        # Unproject to 3D
        points = depth_to_points(depth_ds, fx_ds, fy_ds, cx_ds, cy_ds, pose)

        # Get corresponding colors
        valid = depth_ds > 0
        point_colors = rgb_ds[valid]  # (N, 3)

        N = points.shape[0]
        if N == 0:
            print("Warning: No valid depth points for initialization!")
            return

        # Initialize Gaussian parameters
        self.means = nn.Parameter(points.to(self.device))

        # Scale: based on average distance between nearby points
        avg_dist = 0.01  # default ~1cm
        init_scale = torch.full((N, 3), np.log(avg_dist), device=self.device)
        self.scales = nn.Parameter(init_scale)

        # Quaternions: identity rotation
        init_quats = torch.zeros(N, 4, device=self.device)
        init_quats[:, 0] = 1.0  # w = 1
        self.quats = nn.Parameter(init_quats)

        # Opacities: inverse sigmoid of init_opacity
        init_opacity_logit = torch.logit(
            torch.full((N, 1), self.init_opacity, device=self.device)
        )
        self.opacities = nn.Parameter(init_opacity_logit)

        # Colors: SH DC term from RGB (or just RGB for sh_degree=0)
        if self.sh_degree == 0:
            # Store raw RGB
            init_colors = point_colors.to(self.device)
        else:
            # SH DC coefficient: C0 = color / SH_C0 where SH_C0 = 0.28209479
            SH_C0 = 0.28209479177387814
            dc = point_colors.to(self.device) / SH_C0
            # Zero out higher order SH
            init_colors = torch.zeros(N, self.color_dim, device=self.device)
            init_colors[:, 0] = dc[:, 0]  # R DC
            init_colors[:, self.n_sh_coeffs] = dc[:, 1]  # G DC
            init_colors[:, 2 * self.n_sh_coeffs] = dc[:, 2]  # B DC

        self.colors = nn.Parameter(init_colors)

        # Language features: zeros initially
        self.lang_feats = nn.Parameter(
            torch.zeros(N, self.lang_feat_dim, device=self.device)
        )

        # Contamination tracking: all clean initially
        self.contamination_count = torch.zeros(N, dtype=torch.long, device=self.device)
        self.dynamic_belief = torch.zeros(N, dtype=torch.float32, device=self.device)

        self._num_gaussians = N
        print(f"Initialized {N} Gaussians from depth map")

    def add_gaussians(
        self,
        new_means: torch.Tensor,
        new_colors: torch.Tensor,
        new_scales: torch.Tensor = None,
        new_lang: torch.Tensor = None,
    ) -> int:
        """Add new Gaussians for newly observed regions.

        Args:
            new_means: (M, 3) positions
            new_colors: (M, 3) RGB colors
            new_scales: optional (M, 3) log-space scales. If None, uses default.
            new_lang: optional (M, lang_feat_dim) compressed language features
                observed at each Gaussian's source pixel. If None, features
                start at zero and must be learned via the rendered language
                loss alone — which is far too weak to move them off zero on
                short sequences (observed norms <= 0.43 after full runs).

        Returns:
            Number of Gaussians added
        """
        M = new_means.shape[0]
        if M == 0:
            return 0

        device = self.device

        # Create new parameters
        if new_scales is None:
            new_scales = torch.full((M, 3), np.log(0.01), device=device)
        else:
            new_scales = new_scales.to(device)
        new_quats = torch.zeros(M, 4, device=device)
        new_quats[:, 0] = 1.0
        new_opacities = torch.logit(
            torch.full((M, 1), self.init_opacity, device=device)
        )

        if self.sh_degree == 0:
            new_color_params = new_colors.to(device)
        else:
            SH_C0 = 0.28209479177387814
            dc = new_colors.to(device) / SH_C0
            new_color_params = torch.zeros(M, self.color_dim, device=device)
            new_color_params[:, 0] = dc[:, 0]
            new_color_params[:, self.n_sh_coeffs] = dc[:, 1]
            new_color_params[:, 2 * self.n_sh_coeffs] = dc[:, 2]

        if new_lang is None:
            new_lang = torch.zeros(M, self.lang_feat_dim, device=device)
        else:
            new_lang = new_lang.to(device)

        # Concatenate with existing
        self.means = nn.Parameter(torch.cat([self.means.data, new_means.to(device)]))
        self.scales = nn.Parameter(torch.cat([self.scales.data, new_scales]))
        self.quats = nn.Parameter(torch.cat([self.quats.data, new_quats]))
        self.opacities = nn.Parameter(torch.cat([self.opacities.data, new_opacities]))
        self.colors = nn.Parameter(torch.cat([self.colors.data, new_color_params]))
        self.lang_feats = nn.Parameter(torch.cat([self.lang_feats.data, new_lang]))

        # Extend contamination tracking
        self.contamination_count = torch.cat([
            self.contamination_count,
            torch.zeros(M, dtype=torch.long, device=device),
        ])
        self.dynamic_belief = torch.cat([
            self.dynamic_belief,
            torch.zeros(M, dtype=torch.float32, device=device),
        ])

        self._num_gaussians += M
        return M

    def prune(self, mask: torch.Tensor) -> int:
        """Remove Gaussians where mask is True.

        Args:
            mask: (N,) boolean tensor, True = remove

        Returns:
            Number of Gaussians removed
        """
        keep = ~mask
        n_removed = mask.sum().item()

        self.means = nn.Parameter(self.means.data[keep])
        self.scales = nn.Parameter(self.scales.data[keep])
        self.quats = nn.Parameter(self.quats.data[keep])
        self.opacities = nn.Parameter(self.opacities.data[keep])
        self.colors = nn.Parameter(self.colors.data[keep])
        self.lang_feats = nn.Parameter(self.lang_feats.data[keep])
        self.contamination_count = self.contamination_count[keep]
        self.dynamic_belief = self.dynamic_belief[keep]

        self._num_gaussians -= n_removed
        return n_removed

    def densify(self, grad_norms: torch.Tensor, grad_thresh: float, max_scale: float = 0.05) -> dict:
        """Split large Gaussians and clone small ones based on position gradients.

        Args:
            grad_norms: (N,) per-Gaussian gradient magnitude of 2D means
            grad_thresh: threshold above which to densify
            max_scale: Gaussians with max activated scale > this are split, others cloned

        Returns:
            dict with counts: {"split": int, "cloned": int}
        """
        if self._num_gaussians == 0 or grad_norms is None:
            return {"split": 0, "cloned": 0}

        with torch.no_grad():
            mask = grad_norms > grad_thresh
            if mask.sum() == 0:
                return {"split": 0, "cloned": 0}

            activated_scales = torch.exp(self.scales.data)  # (N, 3)
            max_scales = activated_scales.max(dim=-1).values  # (N,)

            # Split: large Gaussians with high gradient
            split_mask = mask & (max_scales > max_scale)
            # Clone: small Gaussians with high gradient
            clone_mask = mask & ~split_mask

            n_split = int(split_mask.sum().item())
            n_clone = int(clone_mask.sum().item())

            new_means_list = [self.means.data]
            new_scales_list = [self.scales.data]
            new_quats_list = [self.quats.data]
            new_opacities_list = [self.opacities.data]
            new_colors_list = [self.colors.data]
            new_lang_list = [self.lang_feats.data]

            # --- CLONE: duplicate small Gaussians as-is ---
            if n_clone > 0:
                new_means_list.append(self.means.data[clone_mask])
                new_scales_list.append(self.scales.data[clone_mask])
                new_quats_list.append(self.quats.data[clone_mask])
                new_opacities_list.append(self.opacities.data[clone_mask])
                new_colors_list.append(self.colors.data[clone_mask])
                new_lang_list.append(self.lang_feats.data[clone_mask])

            # --- SPLIT: replace large Gaussians with 2 smaller children ---
            if n_split > 0:
                parent_means = self.means.data[split_mask]
                parent_scales = self.scales.data[split_mask]  # log-space
                parent_quats = self.quats.data[split_mask]
                parent_opacities = self.opacities.data[split_mask]
                parent_colors = self.colors.data[split_mask]
                parent_lang = self.lang_feats.data[split_mask]

                # Offset children by sampling from parent's scale
                stds = activated_scales[split_mask]  # (M, 3)
                offsets = torch.randn_like(parent_means) * stds
                child1_means = parent_means + offsets
                child2_means = parent_means - offsets

                # Reduce scale (divide by 1.6 in activated space = subtract log(1.6) in log space)
                shrink = np.log(1.6)
                child_scales = parent_scales - shrink

                # Both children get same rotation, opacity, color, lang
                new_means_list.extend([child1_means, child2_means])
                new_scales_list.extend([child_scales, child_scales])
                new_quats_list.extend([parent_quats, parent_quats])
                new_opacities_list.extend([parent_opacities, parent_opacities])
                new_colors_list.extend([parent_colors, parent_colors])
                new_lang_list.extend([parent_lang, parent_lang])

            # Concatenate all
            all_means = torch.cat(new_means_list)
            all_scales = torch.cat(new_scales_list)
            all_quats = torch.cat(new_quats_list)
            all_opacities = torch.cat(new_opacities_list)
            all_colors = torch.cat(new_colors_list)
            all_lang = torch.cat(new_lang_list)

            # Remove original split parents (they've been replaced by children)
            if n_split > 0:
                keep = ~split_mask
                # Build index: keep originals (minus split parents) + clones + split children
                n_orig = self._num_gaussians
                keep_indices = torch.where(keep)[0]
                clone_indices = torch.arange(n_orig, n_orig + n_clone, device=self.device)
                split_indices = torch.arange(n_orig + n_clone, len(all_means), device=self.device)
                final_indices = torch.cat([keep_indices, clone_indices, split_indices])

                all_means = all_means[final_indices]
                all_scales = all_scales[final_indices]
                all_quats = all_quats[final_indices]
                all_opacities = all_opacities[final_indices]
                all_colors = all_colors[final_indices]
                all_lang = all_lang[final_indices]

            # Extend contamination counts and dynamic belief for clones/splits
            contam_list = [self.contamination_count]
            belief_list = [self.dynamic_belief]
            if n_clone > 0:
                contam_list.append(self.contamination_count[clone_mask])
                belief_list.append(self.dynamic_belief[clone_mask])
            if n_split > 0:
                parent_contam = self.contamination_count[split_mask]
                parent_belief = self.dynamic_belief[split_mask]
                contam_list.extend([parent_contam, parent_contam])
                belief_list.extend([parent_belief, parent_belief])
            all_contam = torch.cat(contam_list)
            all_belief = torch.cat(belief_list)
            if n_split > 0:
                all_contam = all_contam[final_indices]
                all_belief = all_belief[final_indices]

            self.means = nn.Parameter(all_means)
            self.scales = nn.Parameter(all_scales)
            self.quats = nn.Parameter(all_quats)
            self.opacities = nn.Parameter(all_opacities)
            self.colors = nn.Parameter(all_colors)
            self.lang_feats = nn.Parameter(all_lang)
            self.contamination_count = all_contam
            self.dynamic_belief = all_belief
            self._num_gaussians = len(all_means)

        return {"split": n_split, "cloned": n_clone}

    def update_dynamic_belief(
        self,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        mask: torch.Tensor,
        increase: float = 0.2,
        decay: float = 0.07,
        depth: torch.Tensor = None,
        depth_margin: float = 0.15,
    ) -> None:
        """Update per-Gaussian dynamic belief using the current frame's mask.

        Gaussians projecting into dynamic regions have their belief increased.
        All beliefs decay each frame so briefly occluded Gaussians recover.

        Args:
            viewmat: (4, 4) world-to-camera
            K: (3, 3) intrinsics
            width, height: image dimensions
            mask: (H, W) float, 1=static, 0=dynamic
            increase: belief increment for Gaussians in dynamic regions
            decay: belief decay per frame for all Gaussians
            depth: optional (H, W) observed depth in meters. When given, only
                Gaussians at or in front of the observed surface are flagged;
                static background occluded BEHIND a dynamic object (z >
                observed + depth_margin) is left alone. Without this test,
                a person walking across the scene bumps belief on every wall
                and floor Gaussian they pass in front of.
            depth_margin: slack in meters for the occlusion test
        """
        if self._num_gaussians == 0:
            return

        with torch.no_grad():
            # Decay all beliefs toward 0
            self.dynamic_belief = (self.dynamic_belief - decay).clamp(min=0.0)

            # Project means to 2D
            R = viewmat[:3, :3]
            t = viewmat[:3, 3]
            means_cam = self.means.data @ R.T + t.unsqueeze(0)
            z = means_cam[:, 2]

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = (means_cam[:, 0] * fx / z + cx).long()
            v = (means_cam[:, 1] * fy / z + cy).long()

            valid = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if valid.sum() == 0:
                return

            mask_dev = mask.to(self.device) if mask.device != self.device else mask
            is_dynamic = mask_dev[v[valid], u[valid]] < 0.5

            # Occlusion test: don't flag static background behind the object
            if depth is not None:
                depth_dev = depth.to(self.device) if depth.device != self.device else depth
                d_obs = depth_dev[v[valid], u[valid]]
                behind = (d_obs > 0) & (z[valid] > d_obs + depth_margin)
                is_dynamic = is_dynamic & ~behind

            dyn_indices = torch.where(valid)[0][is_dynamic]

            if dyn_indices.shape[0] > 0:
                self.dynamic_belief[dyn_indices] = (
                    self.dynamic_belief[dyn_indices] + increase
                ).clamp(max=1.0)

    def render_belief_mask(
        self,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        belief_thresh: float = 0.5,
        dilation: int = 15,
    ) -> torch.Tensor:
        """Project high-dynamic-belief Gaussians into a view as a mask.

        The language prior deposits evidence into `dynamic_belief` sparsely
        (extraction keyframes); this reprojection carries that evidence to
        EVERY frame as a dense dynamic mask at the cost of one point
        projection. Returns (H, W) bool, True = dynamic.
        """
        mask = torch.zeros(height, width, dtype=torch.bool, device=self.device)
        if self._num_gaussians == 0:
            return mask
        with torch.no_grad():
            hot = self.dynamic_belief > belief_thresh
            if not hot.any():
                return mask
            means = self.means.data[hot]
            R = viewmat[:3, :3]
            t = viewmat[:3, 3]
            means_cam = means @ R.T + t.unsqueeze(0)
            z = means_cam[:, 2]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = (means_cam[:, 0] * fx / z + cx).long()
            v = (means_cam[:, 1] * fy / z + cy).long()
            valid = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not valid.any():
                return mask
            mask[v[valid], u[valid]] = True
            if dilation > 1:
                m = mask.float().unsqueeze(0).unsqueeze(0)
                m = torch.nn.functional.max_pool2d(
                    m, kernel_size=dilation, stride=1, padding=dilation // 2)
                mask = m.squeeze(0).squeeze(0) > 0.5
        return mask

    def seed_lang_feats_from_map(
        self,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        lang_map: torch.Tensor,
        mask: torch.Tensor = None,
        depth: torch.Tensor = None,
        min_norm: float = 0.1,
        depth_margin: float = 0.15,
    ) -> int:
        """Opportunistically seed unseeded Gaussians from a keyframe feature map.

        Gaussians whose lang_feats are still near zero (never seeded at
        creation — e.g. the first-frame map, or keyframes without an
        extraction) get the compressed feature at the pixel they project to,
        provided that pixel is static AND the Gaussian is at the observed
        surface (|z - depth| <= depth_margin), so occluded background never
        inherits foreground features. Call only after the autoencoder is
        frozen so all seeded features share one latent space.

        Args:
            viewmat: (4, 4) world-to-camera
            K: (3, 3) intrinsics
            width, height: image dimensions
            lang_map: (H, W, lang_feat_dim) compressed feature map
            mask: optional (H, W) float, 1=static, 0=dynamic
            depth: optional (H, W) observed depth in meters
            min_norm: Gaussians with feature norm below this are candidates
            depth_margin: surface-agreement slack in meters

        Returns:
            Number of Gaussians seeded
        """
        if self._num_gaussians == 0:
            return 0

        with torch.no_grad():
            unseeded = self.lang_feats.data.norm(dim=-1) < min_norm
            if not unseeded.any():
                return 0

            R = viewmat[:3, :3]
            t = viewmat[:3, 3]
            means_cam = self.means.data @ R.T + t.unsqueeze(0)
            z = means_cam[:, 2]

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = (means_cam[:, 0] * fx / z + cx).long()
            v = (means_cam[:, 1] * fy / z + cy).long()

            valid = (unseeded & (z > 0.01)
                     & (u >= 0) & (u < width) & (v >= 0) & (v < height))
            if not valid.any():
                return 0

            u_v, v_v, z_v = u[valid], v[valid], z[valid]
            keep = torch.ones_like(z_v, dtype=torch.bool)

            if mask is not None:
                mask_dev = mask.to(self.device) if mask.device != self.device else mask
                keep &= mask_dev[v_v, u_v] > 0.5

            if depth is not None:
                depth_dev = depth.to(self.device) if depth.device != self.device else depth
                d_obs = depth_dev[v_v, u_v]
                keep &= (d_obs > 0) & ((z_v - d_obs).abs() <= depth_margin)

            if not keep.any():
                return 0

            idx = torch.where(valid)[0][keep]
            lang_dev = lang_map.to(self.device) if lang_map.device != self.device else lang_map
            self.lang_feats.data[idx] = lang_dev[v_v[keep], u_v[keep]]
            return int(idx.shape[0])

    def get_activated_params(self, suppress_dynamic: bool = True) -> dict:
        """Get activated (post-nonlinearity) Gaussian parameters.

        Args:
            suppress_dynamic: if True, scale opacity by (1 - dynamic_belief)

        Returns:
            dict with activated means, scales (exp), quats (normalized),
            opacities (sigmoid), colors, lang_feats
        """
        opacities = torch.sigmoid(self.opacities)
        if suppress_dynamic and self.dynamic_belief.shape[0] == opacities.shape[0]:
            belief = self.dynamic_belief.unsqueeze(-1)  # (N, 1)
            if belief.any():
                opacities = opacities * (1.0 - belief)

        return {
            "means": self.means,
            "scales": torch.exp(self.scales),
            "quats": torch.nn.functional.normalize(self.quats, dim=-1),
            "opacities": opacities,
            "colors": self.colors if self.sh_degree == 0 else self.colors,
            "lang_feats": self.lang_feats,
        }

    def state_dict_compact(self) -> dict:
        """Save a compact state dict for checkpointing."""
        return {
            "means": self.means.data.cpu(),
            "scales": self.scales.data.cpu(),
            "quats": self.quats.data.cpu(),
            "opacities": self.opacities.data.cpu(),
            "colors": self.colors.data.cpu(),
            "lang_feats": self.lang_feats.data.cpu(),
            "contamination_count": self.contamination_count.cpu(),
            "dynamic_belief": self.dynamic_belief.cpu(),
            "sh_degree": self.sh_degree,
            "lang_feat_dim": self.lang_feat_dim,
            "num_gaussians": self._num_gaussians,
        }

    def load_state_dict_compact(self, state: dict) -> None:
        """Load from compact state dict."""
        self.means = nn.Parameter(state["means"].to(self.device))
        self.scales = nn.Parameter(state["scales"].to(self.device))
        self.quats = nn.Parameter(state["quats"].to(self.device))
        self.opacities = nn.Parameter(state["opacities"].to(self.device))
        self.colors = nn.Parameter(state["colors"].to(self.device))
        self.lang_feats = nn.Parameter(state["lang_feats"].to(self.device))
        self.contamination_count = state.get(
            "contamination_count",
            torch.zeros(state["num_gaussians"], dtype=torch.long, device=self.device),
        ).to(self.device)
        self.dynamic_belief = state.get(
            "dynamic_belief",
            torch.zeros(state["num_gaussians"], dtype=torch.float32, device=self.device),
        ).to(self.device)
        self._num_gaussians = state["num_gaussians"]

    def mark_contaminated(
        self,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        mask: torch.Tensor,
        depth: torch.Tensor = None,
        depth_margin: float = 0.15,
    ) -> int:
        """Increment contamination count for Gaussians projecting into dynamic pixels.

        Projects all Gaussian means to 2D, checks which fall inside dynamic
        regions of the mask, and increments their contamination count.

        Args:
            viewmat: (4, 4) world-to-camera transform
            K: (3, 3) camera intrinsics
            width, height: image dimensions
            mask: (H, W) float — 1=static, 0=dynamic
            depth: optional (H, W) observed depth in meters. When given, only
                Gaussians at or in front of the observed surface are counted;
                occluded static background (z > observed + depth_margin) is
                spared. Without it, cleanup deletes the wall/floor everywhere
                the dynamic object has passed in front.
            depth_margin: slack in meters for the occlusion test

        Returns:
            Number of Gaussians marked as contaminated
        """
        if self._num_gaussians == 0:
            return 0

        with torch.no_grad():
            # Project Gaussian means to 2D pixel coordinates
            R = viewmat[:3, :3]
            t = viewmat[:3, 3]
            means_cam = self.means.data @ R.T + t.unsqueeze(0)  # (N, 3)
            z = means_cam[:, 2]

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = (means_cam[:, 0] * fx / z + cx).long()
            v = (means_cam[:, 1] * fy / z + cy).long()

            # Bounds check
            valid = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

            if valid.sum() == 0:
                return 0

            u_valid = u[valid]
            v_valid = v[valid]

            # Check which valid Gaussians project into dynamic regions
            mask_dev = mask.to(self.device) if mask.device != self.device else mask
            is_dynamic = mask_dev[v_valid, u_valid] < 0.5

            # Occlusion test: don't count static background behind the object
            if depth is not None:
                depth_dev = depth.to(self.device) if depth.device != self.device else depth
                d_obs = depth_dev[v_valid, u_valid]
                behind = (d_obs > 0) & (z[valid] > d_obs + depth_margin)
                is_dynamic = is_dynamic & ~behind

            contaminated_indices = torch.where(valid)[0][is_dynamic]

            n_contaminated = contaminated_indices.shape[0]
            if n_contaminated > 0:
                self.contamination_count[contaminated_indices] += 1

            return n_contaminated

    def cleanup_contaminated(self, threshold: int = 3) -> int:
        """Remove Gaussians that have been repeatedly contaminated by dynamic objects.

        Args:
            threshold: minimum contamination count to trigger removal

        Returns:
            Number of Gaussians removed
        """
        if self._num_gaussians == 0:
            return 0

        prune_mask = self.contamination_count >= threshold
        n = prune_mask.sum().item()
        if n > 0:
            self.prune(prune_mask)
        return n
