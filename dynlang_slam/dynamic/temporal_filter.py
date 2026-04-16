"""Temporal consistency filter for dynamic object masks.

Suppresses false positive detections by requiring a pixel to be flagged
as dynamic in at least `min_detections` out of the last `window_size`
frames. Also applies morphological dilation to cover boundary pixels.
"""

from collections import deque

import torch
import torch.nn.functional as F


class TemporalFilter:
    """Sliding-window temporal filter with mask dilation."""

    def __init__(
        self,
        window_size: int = 3,
        min_detections: int = 2,
        dilation_kernel: int = 5,
    ):
        """Initialize temporal filter.

        Args:
            window_size: number of frames in the sliding window
            min_detections: minimum times a pixel must be flagged dynamic
                in the window to be confirmed as dynamic
            dilation_kernel: square kernel size for mask dilation (must be odd)
        """
        self.window_size = window_size
        self.min_detections = min_detections
        self.dilation_kernel = dilation_kernel
        self._history: deque[torch.Tensor] = deque(maxlen=window_size)

    def update(self, raw_dynamic_mask: torch.Tensor) -> torch.Tensor:
        """Add a new detection mask and return the temporally-filtered result.

        Args:
            raw_dynamic_mask: (H, W) bool tensor — True = dynamic pixel

        Returns:
            static_mask: (H, W) float tensor — 1.0 = static (use), 0.0 = dynamic (skip)
            Matches the mask convention in tracker.py (1=valid, 0=ignore).
        """
        # Store on CPU to save GPU memory
        self._history.append(raw_dynamic_mask.cpu().bool())

        # Stack history and count detections per pixel
        # Shape: (window, H, W)
        stacked = torch.stack(list(self._history), dim=0)
        detection_count = stacked.float().sum(dim=0)  # (H, W)

        # Confirm dynamic if detected enough times
        dynamic_confirmed = detection_count >= self.min_detections

        # Dilate to cover boundary pixels
        if self.dilation_kernel > 1 and dynamic_confirmed.any():
            dynamic_confirmed = self._dilate_mask(dynamic_confirmed)

        # Invert: 1=static (keep), 0=dynamic (skip)
        static_mask = (~dynamic_confirmed).float()

        return static_mask

    def _dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Dilate a binary mask using max-pooling (GPU-friendly).

        F.max_pool2d with stride=1 is equivalent to morphological dilation
        with a square structuring element.

        Args:
            mask: (H, W) bool tensor

        Returns:
            (H, W) bool tensor — dilated mask
        """
        k = self.dilation_kernel
        pad = k // 2

        # max_pool2d needs (N, C, H, W) float input
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(mask_4d, kernel_size=k, stride=1, padding=pad)

        return dilated.squeeze(0).squeeze(0) > 0.5

    def reset(self) -> None:
        """Clear the history buffer (e.g., on scene change)."""
        self._history.clear()

    @property
    def has_history(self) -> bool:
        """Whether the filter has any frames in its history."""
        return len(self._history) > 0

    @property
    def history_length(self) -> int:
        """Number of frames currently in the sliding window."""
        return len(self._history)
