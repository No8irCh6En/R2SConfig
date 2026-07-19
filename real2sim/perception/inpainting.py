"""
Background inpainting via LaMa (Large Mask Inpainting).

Recovers occluded regions behind objects using iterative mask expansion
and sequential inpainting for cleaner background generation.
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import tempfile


class LaMaInpainter:
    """Wrapper for LaMa inpainting with iterative mask expansion support."""

    def __init__(self, model_path: str = "advimman/lama", device: str = "cuda", pad_to: int = 8):
        self.device = torch.device(device)
        self.pad_to = pad_to
        self.model_available = False
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config

            self.model = ModelManager(name="lama", device=self.device)
            self.model_available = True
            self._backend = "lama_cleaner"
        except ImportError:
            try:
                from simple_lama_inpainting import SimpleLama

                self.model = SimpleLama(device=str(self.device))
                self.model_available = True
                self._backend = "simple_lama"
            except ImportError:
                print(
                    "[LaMa] lama-cleaner or simple-lama-inpainting not found; install with:\n"
                    "  pip install lama-cleaner   # or\n"
                    "  pip install simple-lama-inpainting"
                )

    def inpaint(
        self, image: Image.Image, mask: torch.Tensor, refine_iters: int = 0
    ) -> Image.Image:
        """Inpaint masked regions.

        Args:
            image: PIL RGB image
            mask: (H, W) binary tensor, True = region to inpaint
            refine_iters: additional pass iterations with slightly expanded mask

        Returns:
            Inpainted PIL image
        """
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np)

        result = self._run_inpaint(image, mask_img)

        for _ in range(refine_iters):
            from ..viz.utils import dilate_mask

            mask = dilate_mask(mask, kernel_size=5, iters=1)
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np)
            result = self._run_inpaint(result, mask_img)

        return result

    def _run_inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if not self.model_available:
            raise RuntimeError("LaMa model not available; install lama-cleaner or simple-lama-inpainting")

        if self._backend == "lama_cleaner":
            return self._inpaint_lama_cleaner(image, mask)
        return self._inpaint_simple_lama(image, mask)

    def _inpaint_lama_cleaner(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import cv2

        img_np = np.array(image)
        mask_np = np.array(mask)
        result = self.model(img_np, mask_nm=mask_np)
        return Image.fromarray(result)

    def _inpaint_simple_lama(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        result = self.model(image, mask)
        return result


def inpaint_with_sequential_fill(
    inpainter: LaMaInpainter,
    image: Image.Image,
    masks: torch.Tensor,  # (N, H, W)
    border_expand: int = 5,
) -> Image.Image:
    """Sequentially inpaint multiple object masks with border expansion.

    Each object is inpainted one at a time, with the mask slightly expanded
    to include the transition boundary between object and background.
    """
    result = image.copy()
    for mask in masks:
        from ..viz.utils import dilate_mask

        dilated = dilate_mask(mask, kernel_size=border_expand, iters=1)
        result = inpainter.inpaint(result, dilated)
    return result
