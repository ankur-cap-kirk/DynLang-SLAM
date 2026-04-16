"""Loss functions for DynLang-SLAM."""

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """L1 loss, optionally masked."""
    loss = torch.abs(pred - target)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def huber_loss(
    pred: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor = None, delta: float = 0.1,
) -> torch.Tensor:
    """Huber loss, optionally masked. More robust to depth outliers than L1.

    Quadratic for |error| < delta, linear for |error| >= delta.
    """
    loss = F.huber_loss(pred, target, reduction='none', delta=delta)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def ssim_loss(
    pred: torch.Tensor, target: torch.Tensor,
    window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2,
) -> torch.Tensor:
    """Structural Similarity Index loss (1 - SSIM).

    Args:
        pred: (H, W, C) or (1, C, H, W) predicted image
        target: same shape as pred

    Returns:
        Scalar loss (1 - SSIM), lower is better
    """
    # Reshape to (1, C, H, W) for conv2d
    if pred.dim() == 3:
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)

    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, C, pred.device)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


def _gaussian_kernel(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.outer(g)
    kernel = kernel_2d.expand(channels, 1, size, size).contiguous()
    return kernel


def language_loss(
    pred_lang: torch.Tensor,
    gt_lang: torch.Tensor,
    alpha: torch.Tensor = None,
) -> torch.Tensor:
    """Language feature loss: L1 + cosine distance.

    Args:
        pred_lang: (H, W, D) rendered language features
        gt_lang: (H, W, D) target language features (autoencoder-compressed CLIP)
        alpha: (H, W, 1) or (H, W) rendered alpha, used to mask unmapped regions

    Returns:
        Scalar language loss
    """
    if alpha is not None:
        if alpha.dim() == 3:
            alpha = alpha.squeeze(-1)
        # Only supervise where map has coverage
        valid = (alpha > 0.5).float()
        if valid.sum() < 1:
            return torch.tensor(0.0, device=pred_lang.device)

        # L1 loss (masked)
        diff = torch.abs(pred_lang - gt_lang)  # (H, W, D)
        loss_l1 = (diff * valid.unsqueeze(-1)).sum() / (valid.sum() * pred_lang.shape[-1] + 1e-8)

        # Cosine distance (masked)
        cos_sim = F.cosine_similarity(pred_lang, gt_lang, dim=-1)  # (H, W)
        loss_cos = ((1.0 - cos_sim) * valid).sum() / (valid.sum() + 1e-8)
    else:
        loss_l1 = F.l1_loss(pred_lang, gt_lang)
        cos_sim = F.cosine_similarity(pred_lang, gt_lang, dim=-1)
        loss_cos = (1.0 - cos_sim).mean()

    return loss_l1 + loss_cos


def compute_soft_dynamic_weights(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    quantile: float = 0.90,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """Compute soft per-pixel weights from photometric residuals (DGS-SLAM).

    Pixels with high residual (likely dynamic) get down-weighted smoothly
    instead of being removed entirely, preserving tracking signal.

    Args:
        pred_rgb: (H, W, 3) rendered RGB
        gt_rgb: (H, W, 3) ground truth RGB
        quantile: residual percentile above which pixels are down-weighted
        sharpness: sigmoid steepness (higher = sharper transition)

    Returns:
        (H, W) float weights in [0, 1], 1 = fully trusted
    """
    with torch.no_grad():
        residual = (pred_rgb - gt_rgb).abs().mean(dim=-1)  # (H, W)
        flat = residual.reshape(-1)
        k = int(flat.shape[0] * quantile)
        threshold = flat.kthvalue(k).values.clamp(min=1e-4)
        weights = 1.0 - torch.sigmoid(sharpness * (residual - threshold) / threshold)
    return weights


def compute_losses(
    rendered: dict,
    gt_rgb: torch.Tensor,
    gt_depth: torch.Tensor,
    weights: dict,
    mask: torch.Tensor = None,
    use_soft_dynamic: bool = False,
    use_huber: bool = True,
    huber_rgb_delta: float = 0.05,
    huber_depth_delta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute combined tracking/mapping loss.

    Args:
        rendered: dict from renderer with 'rgb', 'depth', 'alpha'
        gt_rgb: (H, W, 3) ground truth RGB [0, 1]
        gt_depth: (H, W) or (H, W, 1) ground truth depth in meters
        weights: dict with 'rgb_weight', 'depth_weight', 'ssim_weight'
        mask: optional (H, W) binary mask (1 = valid, 0 = ignore)
        use_soft_dynamic: if True, apply residual-based soft weighting

    Returns:
        total_loss, loss_dict with individual components
    """
    pred_rgb = rendered["rgb"]       # (H, W, 3)
    pred_depth = rendered["depth"]   # (H, W, 1)
    alpha = rendered["alpha"]        # (H, W, 1)

    # Ensure gt_depth is 2D (H, W) — handle both (1, H, W) and (H, W, 1)
    while gt_depth.dim() > 2:
        gt_depth = gt_depth.squeeze(-1) if gt_depth.shape[-1] == 1 else gt_depth.squeeze(0)
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(-1)  # (H, W)
    if alpha.dim() == 3:
        alpha_2d = alpha.squeeze(-1)  # (H, W)
    else:
        alpha_2d = alpha

    # Soft dynamic weighting: down-weight high-residual pixels
    soft_w = None
    if use_soft_dynamic:
        soft_w = compute_soft_dynamic_weights(pred_rgb, gt_rgb)

    # Depth validity mask: only where GT depth is valid AND map has Gaussians
    depth_valid = ((gt_depth > 0) & (alpha_2d > 0.5)).float()
    if mask is not None:
        depth_valid = depth_valid * mask
    if soft_w is not None:
        depth_valid = depth_valid * soft_w

    # RGB loss: also weight by alpha to reduce influence of unmapped regions
    alpha_rgb = alpha_2d.unsqueeze(-1).expand_as(pred_rgb)
    rgb_mask = alpha_rgb
    if mask is not None:
        rgb_mask = rgb_mask * mask.unsqueeze(-1).expand_as(pred_rgb)
    if soft_w is not None:
        rgb_mask = rgb_mask * soft_w.unsqueeze(-1).expand_as(pred_rgb)
    # RGB loss: Huber gives L2-like gradient near optimum (better refinement)
    # and L1-like robustness for outliers (missed dynamics, occlusions)
    if use_huber:
        loss_rgb = huber_loss(pred_rgb, gt_rgb, rgb_mask, delta=huber_rgb_delta)
    else:
        loss_rgb = l1_loss(pred_rgb, gt_rgb, rgb_mask)

    # Depth loss (only where GT depth is valid AND map has coverage)
    if use_huber:
        loss_depth = huber_loss(pred_depth, gt_depth, depth_valid, delta=huber_depth_delta)
    else:
        loss_depth = l1_loss(pred_depth, gt_depth, depth_valid)

    # SSIM loss
    loss_ssim = ssim_loss(pred_rgb, gt_rgb)

    # Weighted total
    total = (
        weights.get("rgb_weight", 0.5) * loss_rgb
        + weights.get("depth_weight", 1.0) * loss_depth
        + weights.get("ssim_weight", 0.2) * loss_ssim
    )

    loss_dict = {
        "rgb": loss_rgb.item(),
        "depth": loss_depth.item(),
        "ssim": loss_ssim.item(),
        "total": total.item(),
    }
    return total, loss_dict
