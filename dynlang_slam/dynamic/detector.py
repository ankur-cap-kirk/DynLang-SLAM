"""YOLOv8 dynamic object detector for DynLang-SLAM.

Detects dynamic objects (people, cars, etc.) using YOLOv8 instance segmentation.
Returns per-pixel binary masks that are used to exclude dynamic regions from
tracking and mapping, preventing map contamination.

GPU offloading: model resides on CPU, moved to GPU only during detect(),
then immediately moved back — same pattern as CLIP/SAM2 extractors.
"""

import torch
import torch.nn.functional as F
import numpy as np


class DynamicDetector:
    """YOLOv8 instance segmentation wrapper with GPU offloading."""

    # Default COCO dynamic classes
    DEFAULT_DYNAMIC_CLASSES = [
        0,   # person
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        14,  # bird
        15,  # cat
        16,  # dog
        17,  # horse
    ]

    def __init__(
        self,
        model_name: str = "yolov8x-seg",
        confidence_thresh: float = 0.5,
        dynamic_classes: list[int] = None,
        device: str = "cuda",
    ):
        """Initialize YOLOv8 instance segmentation model.

        Args:
            model_name: YOLOv8 variant (yolov8n-seg, yolov8s-seg, yolov8x-seg)
            confidence_thresh: minimum detection confidence
            dynamic_classes: COCO class IDs to treat as dynamic
            device: target GPU device for inference
        """
        from ultralytics import YOLO

        self.device = device
        self.confidence_thresh = confidence_thresh
        self.dynamic_classes = set(
            dynamic_classes if dynamic_classes is not None
            else self.DEFAULT_DYNAMIC_CLASSES
        )

        # Load model directly on target device
        # YOLOv8n/s are small enough to keep on GPU permanently (~6-23MB).
        # Only offload for large models (yolov8x-seg ~136MB) on tight GPU budgets.
        self.model = YOLO(model_name)
        self._keep_on_gpu = "n-seg" in model_name or "s-seg" in model_name
        self.model_name = model_name
        if self._keep_on_gpu:
            self.model.to(self.device)
        print(f"  YOLOv8 loaded: {model_name} "
              f"(dynamic classes: {sorted(self.dynamic_classes)}, "
              f"persistent_gpu={self._keep_on_gpu})")

    @torch.no_grad()
    def detect(self, rgb: torch.Tensor) -> list[dict]:
        """Run YOLOv8 instance segmentation on a single frame.

        Args:
            rgb: (3, H, W) tensor [0, 1] on any device, or (H, W, 3) numpy uint8

        Returns:
            list of dicts, each with:
                'class_id': int (COCO class ID)
                'class_name': str
                'confidence': float
                'mask': (H, W) bool tensor on CPU
                'bbox': (4,) tensor [x1, y1, x2, y2]
        """
        # Convert to numpy uint8 (H, W, 3) — ultralytics expects this
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 3 and rgb.shape[0] == 3:
                rgb_np = (rgb.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        else:
            rgb_np = rgb

        H, W = rgb_np.shape[:2]

        # Move model to GPU if not already there
        if not self._keep_on_gpu:
            self.model.to(self.device)

        # Run inference (verbose=False suppresses per-frame logging)
        results = self.model.predict(
            rgb_np,
            conf=self.confidence_thresh,
            verbose=False,
            device=self.device,
        )

        # Offload back to CPU only for large models
        if not self._keep_on_gpu:
            self.model.to("cpu")
            torch.cuda.empty_cache()

        # Parse results
        detections = []
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            masks_data = result.masks.data  # (N, mask_H, mask_W)
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # Only keep dynamic classes
                if class_id not in self.dynamic_classes:
                    continue

                confidence = boxes.conf[i].item()
                bbox = boxes.xyxy[i].cpu()

                # Resize mask to original image resolution (nearest-neighbor)
                mask = masks_data[i].unsqueeze(0).unsqueeze(0).float()
                mask = F.interpolate(
                    mask, size=(H, W), mode="nearest"
                ).squeeze(0).squeeze(0)
                mask_bool = mask > 0.5

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": confidence,
                    "mask": mask_bool.cpu(),
                    "bbox": bbox,
                })

        return detections

    @torch.no_grad()
    def detect_and_merge(self, rgb: torch.Tensor) -> torch.Tensor:
        """Detect dynamic objects and merge all masks into one binary mask.

        Args:
            rgb: (3, H, W) tensor [0, 1] or (H, W, 3) numpy uint8

        Returns:
            (H, W) bool tensor on CPU — True where any dynamic object detected.
            If no detections, returns all-False tensor.
        """
        # Get image dimensions
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 3 and rgb.shape[0] == 3:
                H, W = rgb.shape[1], rgb.shape[2]
            else:
                H, W = rgb.shape[0], rgb.shape[1]
        else:
            H, W = rgb.shape[:2]

        detections = self.detect(rgb)

        if len(detections) == 0:
            return torch.zeros(H, W, dtype=torch.bool)

        # Merge all detection masks with OR
        merged = torch.zeros(H, W, dtype=torch.bool)
        for det in detections:
            merged = merged | det["mask"]

        return merged
