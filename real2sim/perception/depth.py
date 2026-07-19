"""
Monocular depth estimation via DepthAnything V2.

Produces relative or metric depth maps used for:
- Background geometry initialization
- Depth supervision in the optimization stage
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from typing import Optional


class DepthEstimator:
    """DepthAnything V2 wrapper for monocular depth estimation."""

    def __init__(self, model_size: str = "large", device: str = "cuda"):
        self.device = torch.device(device)
        self.model_size = model_size
        self.model_available = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = f"depth-anything/Depth-Anything-V2-{self.model_size}-hf"
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self.model_available = True
        except Exception as e:
            print(f"[DepthEstimator] DepthAnything V2 unavailable ({e}); falling back to V1...")
            try:
                from transformers import DPTImageProcessor, DPTForDepthEstimation

                model_id = f"LiheYoung/depth-anything-{self.model_size}-hf"
                self.processor = DPTImageProcessor.from_pretrained(model_id)
                self.model = DPTForDepthEstimation.from_pretrained(model_id).to(self.device)
                self.model.eval()
                self.model_available = True
            except Exception as e2:
                print(f"[DepthEstimator] DepthAnything V1 also unavailable ({e2}).")

    @torch.no_grad()
    def estimate(self, image: Image.Image) -> torch.Tensor:
        """Estimate depth map.

        Args:
            image: PIL RGB image

        Returns:
            (H, W) inverse-depth map (higher = closer). Values are relative.
        """
        if not self.model_available:
            return self._fallback_depth(image)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # HuggingFace depth models output predicted_depth
        depth = outputs.predicted_depth
        if depth.ndim == 4:
            depth = depth[:, 0]

        # Resize to original image size
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)
        depth = torch.nn.functional.interpolate(
            depth, size=image.size[::-1], mode="bilinear", align_corners=False
        )[0, 0]

        return depth

    def _fallback_depth(self, image: Image.Image) -> torch.Tensor:
        """Uniform-depth fallback when no model is available."""
        W, H = image.size
        ys = torch.linspace(0, 1, H)[:, None].repeat(1, W)
        depth = 1.0 - ys * 0.5  # slight depth gradient
        return depth

    def estimate_with_mask(
        self, image: Image.Image, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate depth and optionally extract masked region statistics.

        Returns:
            depth: (H, W) full depth map
            depth_masked: (H, W) depth map with non-mask regions zeroed
        """
        depth = self.estimate(image)
        if mask is not None:
            depth_masked = depth * mask.float().to(depth.device)
        else:
            depth_masked = depth
        return depth, depth_masked
