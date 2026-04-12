"""Pixel-aligned CLIP feature map builder for DynLang-SLAM.

Combines SAM masks and CLIP features to create dense per-pixel language
feature maps at 3 semantic scales, following the LangSplat pipeline.
"""

import numpy as np
import torch

from .clip_extractor import CLIPExtractor
from .sam_extractor import SAMExtractor


class FeatureMapBuilder:
    """Builds pixel-aligned CLIP feature maps from SAM masks."""

    def __init__(
        self,
        clip_extractor: CLIPExtractor,
        sam_extractor: SAMExtractor,
        device: str = "cuda",
    ):
        self.clip = clip_extractor
        self.sam = sam_extractor
        self.device = device
        self.feat_dim = clip_extractor.feat_dim  # 768

    def build_feature_maps(
        self, image: np.ndarray
    ) -> dict[str, torch.Tensor]:
        """Build per-pixel CLIP feature maps at all 3 scales.

        Args:
            image: (H, W, 3) uint8 numpy array [0, 255]

        Returns:
            dict with keys "subpart", "part", "whole".
            Each value is a (H, W, 768) tensor of CLIP features on device.
        """
        H, W = image.shape[:2]

        # Get global image feature as fallback for unmapped pixels
        global_feat = self.clip.encode_image(image)  # (768,)

        # Generate masks at all scales
        all_masks = self.sam.generate_masks(image)

        feature_maps = {}
        for scale_name, masks in all_masks.items():
            feat_map = self._build_single_scale(
                image, masks, global_feat, H, W
            )
            feature_maps[scale_name] = feat_map

        return feature_maps

    def build_single_scale(
        self, image: np.ndarray, scale: str = "whole"
    ) -> torch.Tensor:
        """Build feature map for a single scale (for faster testing).

        Args:
            image: (H, W, 3) uint8 numpy array [0, 255]
            scale: one of "subpart", "part", "whole"

        Returns:
            (H, W, 768) tensor of CLIP features
        """
        H, W = image.shape[:2]
        global_feat = self.clip.encode_image(image)
        masks = self.sam.generate_masks_single_scale(image, scale)
        return self._build_single_scale(image, masks, global_feat, H, W)

    def _build_single_scale(
        self,
        image: np.ndarray,
        masks: list[dict],
        global_feat: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Build feature map for one set of masks.

        Each pixel gets the CLIP feature of its highest-IoU mask.
        Unmapped pixels get the global image feature.
        """
        # Initialize with global feature everywhere
        feat_map = global_feat.unsqueeze(0).unsqueeze(0).expand(H, W, -1).clone()
        iou_map = torch.zeros(H, W, device=self.device)

        if len(masks) == 0:
            return feat_map

        # Encode all masks through CLIP
        mask_feats = self.clip.encode_masks(image, masks)

        # Assign features to pixels
        for idx, (mask_info, feat) in enumerate(zip(masks, mask_feats)):
            seg = mask_info["segmentation"]  # (H, W) bool numpy
            iou = mask_info["predicted_iou"]

            seg_tensor = torch.from_numpy(seg).to(self.device)
            update = seg_tensor & (iou > iou_map)

            if update.any():
                feat_map[update] = feat
                iou_map[update] = iou

        return feat_map

    @staticmethod
    def compute_similarity(
        feature_map: torch.Tensor, query_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between feature map and text query.

        Args:
            feature_map: (H, W, D) normalized CLIP features
            query_feat: (D,) normalized CLIP text feature

        Returns:
            (H, W) similarity map in [-1, 1]
        """
        return torch.einsum("hwd,d->hw", feature_map, query_feat)

    @staticmethod
    def compute_relevancy(
        feature_map: torch.Tensor,
        query_feat: torch.Tensor,
        clip_extractor: "CLIPExtractor",
        canonical_phrases: list[str] = None,
    ) -> torch.Tensor:
        """Compute relevancy score using LangSplat's contrastive formula.

        Instead of raw cosine similarity (narrow range), this computes:
            relevancy = exp(sim_query) / (exp(sim_query) + max(exp(sim_canon_i)))

        This pushes scores to [0, 1] with sharper peaks on matching objects.

        Args:
            feature_map: (H, W, D) normalized CLIP features
            query_feat: (D,) normalized CLIP text feature
            clip_extractor: CLIPExtractor for encoding canonical phrases
            canonical_phrases: contrast phrases (default: ["object", "things", "stuff", "texture"])

        Returns:
            (H, W) relevancy map in [0, 1]
        """
        if canonical_phrases is None:
            canonical_phrases = ["object", "things", "stuff", "texture"]

        # Temperature scaling amplifies small differences in cosine similarity
        # CLIP cosine sims are typically in [0.1, 0.3], so we need high temp
        temperature = 50.0

        # Query similarity
        sim_query = torch.einsum("hwd,d->hw", feature_map, query_feat)

        # Canonical phrase similarities
        canon_feats = clip_extractor.encode_texts(canonical_phrases)  # (C, D)
        sim_canons = torch.einsum("hwd,cd->hwc", feature_map, canon_feats)  # (H, W, C)
        max_canon = sim_canons.max(dim=-1).values  # (H, W)

        # Relevancy: softmax between query and best canonical match (with temperature)
        exp_query = torch.exp(temperature * sim_query)
        exp_canon = torch.exp(temperature * max_canon)
        relevancy = exp_query / (exp_query + exp_canon)

        return relevancy

    @staticmethod
    def similarity_to_heatmap(sim_map: torch.Tensor) -> np.ndarray:
        """Convert similarity map to a colorized heatmap for visualization.

        Args:
            sim_map: (H, W) similarity values

        Returns:
            (H, W, 3) uint8 heatmap (blue=low, red=high)
        """
        import matplotlib.cm as cm

        sim_np = sim_map.cpu().numpy()
        # Normalize to [0, 1]
        vmin, vmax = sim_np.min(), sim_np.max()
        if vmax - vmin > 1e-6:
            sim_norm = (sim_np - vmin) / (vmax - vmin)
        else:
            sim_norm = np.zeros_like(sim_np)

        heatmap = cm.jet(sim_norm)[:, :, :3]  # (H, W, 3) float [0, 1]
        return (heatmap * 255).astype(np.uint8)
