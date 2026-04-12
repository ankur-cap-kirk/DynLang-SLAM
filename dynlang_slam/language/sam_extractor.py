"""SAM2 multi-scale mask generation for DynLang-SLAM.

Generates hierarchical masks at 3 semantic scales (subpart, part, whole)
using Meta's SAM2.1 Hiera model, following the LangSplat approach.

SAM2.1 Hiera-T is ~150MB vs SAM ViT-L ~2.4GB, much faster and lighter.
"""

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# Scale configurations: (points_per_side, min_mask_region_area)
SCALE_CONFIGS = {
    "subpart": {"points_per_side": 64, "min_mask_region_area": 100},
    "part":    {"points_per_side": 32, "min_mask_region_area": 1000},
    "whole":   {"points_per_side": 16, "min_mask_region_area": 5000},
}


class SAMExtractor:
    """Generates multi-scale segmentation masks using SAM2."""

    def __init__(
        self,
        model_type: str = "sam2.1_hiera_t",
        checkpoint_path: str = "checkpoints/sam2.1_hiera_tiny.pt",
        device: str = "cuda",
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.8,
    ):
        self.device = device
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh

        # Map model_type to config file
        config_map = {
            "sam2.1_hiera_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        config_file = config_map.get(model_type, "configs/sam2.1/sam2.1_hiera_t.yaml")

        # Build SAM2 model
        sam2 = build_sam2(config_file, ckpt_path=checkpoint_path, device=device)
        self.sam2 = sam2

        # Create generators for each scale
        self.generators = {}
        for scale_name, cfg in SCALE_CONFIGS.items():
            self.generators[scale_name] = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=cfg["points_per_side"],
                min_mask_region_area=cfg["min_mask_region_area"],
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=0,
            )

    def generate_masks(self, image: np.ndarray) -> dict[str, list[dict]]:
        """Generate masks at 3 semantic scales.

        Args:
            image: (H, W, 3) uint8 numpy array [0, 255]

        Returns:
            dict with keys "subpart", "part", "whole".
            Each value is a list of mask dicts with:
                - 'segmentation': (H, W) bool array
                - 'bbox': [x, y, w, h]
                - 'predicted_iou': float
                - 'stability_score': float
                - 'area': int (number of pixels)
        """
        results = {}
        for scale_name, generator in self.generators.items():
            masks = generator.generate(image)
            masks.sort(key=lambda m: m["area"], reverse=True)
            results[scale_name] = masks

        return results

    def generate_masks_single_scale(
        self, image: np.ndarray, scale: str = "whole"
    ) -> list[dict]:
        """Generate masks at a single scale (for faster testing).

        Args:
            image: (H, W, 3) uint8 numpy array [0, 255]
            scale: one of "subpart", "part", "whole"

        Returns:
            list of mask dicts
        """
        return self.generators[scale].generate(image)

    @staticmethod
    def masks_to_segmentation_map(
        masks: list[dict], height: int, width: int
    ) -> np.ndarray:
        """Convert list of masks to a single segmentation map.

        Each pixel gets the ID of the mask with the highest predicted_iou
        that covers it. Pixels with no mask get ID -1.

        Args:
            masks: list of mask dicts from generate_masks
            height, width: image dimensions

        Returns:
            (H, W) int32 array with mask IDs (-1 = no mask)
        """
        seg_map = np.full((height, width), -1, dtype=np.int32)
        iou_map = np.zeros((height, width), dtype=np.float32)

        for idx, mask_info in enumerate(masks):
            seg = mask_info["segmentation"]
            iou = mask_info["predicted_iou"]
            update = seg & (iou > iou_map)
            seg_map[update] = idx
            iou_map[update] = iou

        return seg_map

    @staticmethod
    def visualize_masks(
        image: np.ndarray, masks: list[dict], alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay colored masks on the image for visualization."""
        vis = image.copy().astype(np.float32)
        rng = np.random.RandomState(42)

        for mask_info in masks:
            seg = mask_info["segmentation"]
            color = rng.randint(50, 255, size=3).astype(np.float32)
            vis[seg] = vis[seg] * (1 - alpha) + color * alpha

        return vis.clip(0, 255).astype(np.uint8)
