"""Differentiable Gaussian Splatting renderer using gsplat.

Uses gsplat's optimized CUDA rasterization for high-performance rendering
with full gradient flow through both Gaussian parameters and camera pose.
"""

import torch
import torch.nn as nn
from gsplat import rasterization


class GaussianRenderer(nn.Module):
    """Differentiable Gaussian splatting renderer using gsplat CUDA kernels."""

    def __init__(self, near: float = 0.01, far: float = 100.0, splat_radius: int = 4):
        super().__init__()
        self.near = near
        self.far = far

    def forward(
        self,
        gaussian_map,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        render_lang: bool = False,
        bg_color: torch.Tensor = None,
        downscale: int = 1,
    ) -> dict:
        device = viewmat.device
        H = height // downscale
        W = width // downscale

        if gaussian_map.num_gaussians == 0:
            return self._empty_result(H, W, device, render_lang, gaussian_map)

        if bg_color is None:
            bg_color = torch.zeros(3, device=device)

        params = gaussian_map.get_activated_params()
        means = params["means"]       # (N, 3)
        scales = params["scales"]     # (N, 3)
        quats = params["quats"]       # (N, 4) [w, x, y, z]
        opacities = params["opacities"].squeeze(-1)  # (N,)
        colors = params["colors"]     # (N, 3)

        # Build downscaled intrinsic matrix
        K_ds = K.clone()
        if downscale > 1:
            K_ds[0, :] = K[0, :] / downscale
            K_ds[1, :] = K[1, :] / downscale

        # gsplat expects batched viewmats (C, 4, 4) and Ks (C, 3, 3)
        viewmats_batch = viewmat.unsqueeze(0)  # (1, 4, 4)
        Ks_batch = K_ds.unsqueeze(0)           # (1, 3, 3)

        # Single pass: render RGB + Expected Depth
        # Concatenate depth as extra channel to colors so we get both in one pass
        # Compute per-Gaussian camera-space depth for depth rendering
        R_w2c = viewmat[:3, :3]
        t_w2c = viewmat[:3, 3]
        means_cam = means @ R_w2c.T + t_w2c.unsqueeze(0)
        depths = means_cam[:, 2:3]  # (N, 1) camera-space z

        # Colors with depth channel: (N, 4)
        colors_with_depth = torch.cat([colors, depths], dim=-1)

        rendered, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_with_depth,
            viewmats=viewmats_batch,
            Ks=Ks_batch,
            width=W,
            height=H,
            near_plane=self.near,
            far_plane=self.far,
            render_mode="RGB",  # treats all color channels as generic features
            packed=True,
            sh_degree=None,
            absgrad=True,
        )

        # Split output: (1, H, W, 4) -> RGB (H,W,3) + depth (H,W,1)
        out = rendered[0]  # (H, W, 4)
        rgb_out = out[:, :, :3].clamp(0, 1)
        depth_premul = out[:, :, 3:4]  # alpha-premultiplied depth
        alpha_out = alphas[0]  # (H, W, 1)

        # Normalize depth by alpha to get actual surface depth
        # gsplat composites depth as sum(T_i * alpha_i * z_i), need to divide by alpha
        depth_out = torch.where(
            alpha_out > 1e-4,
            depth_premul / alpha_out.clamp(min=1e-4),
            torch.zeros_like(depth_premul),
        )

        # Apply background
        rgb_out = rgb_out + (1.0 - alpha_out) * bg_color.view(1, 1, 3)

        result = {
            "rgb": rgb_out,
            "depth": depth_out,
            "alpha": alpha_out,
            "meta": meta,
        }

        # Render language features separately if needed
        if render_lang and hasattr(gaussian_map, 'lang_feats') and gaussian_map.lang_feats is not None:
            lang_feats = params.get("lang_feats")
            if lang_feats is not None:
                lang_rendered, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=lang_feats,
                    viewmats=viewmats_batch,
                    Ks=Ks_batch,
                    width=W,
                    height=H,
                    near_plane=self.near,
                    far_plane=self.far,
                    render_mode="RGB",
                    packed=True,
                    sh_degree=None,
                )
                result["lang_feats"] = lang_rendered[0]  # (H, W, D)

        return result

    def render_silhouette(self, gaussian_map, viewmat, K, width, height):
        """Render alpha/silhouette map for map expansion decisions."""
        result = self.forward(gaussian_map, viewmat, K, width, height,
                              render_lang=False, downscale=8)
        alpha = result["alpha"].squeeze(-1)
        if alpha.shape[0] != height or alpha.shape[1] != width:
            alpha = torch.nn.functional.interpolate(
                alpha.unsqueeze(0).unsqueeze(0),
                size=(height, width), mode="nearest"
            ).squeeze(0).squeeze(0)
        return alpha

    def _empty_result(self, H, W, device, render_lang, gaussian_map):
        result = {
            "rgb": torch.zeros(H, W, 3, device=device),
            "depth": torch.zeros(H, W, 1, device=device),
            "alpha": torch.zeros(H, W, 1, device=device),
        }
        if render_lang:
            result["lang_feats"] = torch.zeros(
                H, W, gaussian_map.lang_feat_dim, device=device
            )
        return result
