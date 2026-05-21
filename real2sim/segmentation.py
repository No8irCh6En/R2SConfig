"""
Text-driven instance segmentation via SAM3 (primary) or Grounding-DINO + SAM2 (fallback).

Primary path (SAM3):   text prompt → SAM3 set_text_prompt() → masks (demo.py style)
Fallback path (DINO):   text prompt → Grounding-DINO boxes → SAM2 refinement → masks
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    masks: torch.Tensor   # (N, H, W) binary masks
    boxes: torch.Tensor   # (N, 4) xyxy boxes (in pixel coords)
    scores: torch.Tensor  # (N,) confidence scores
    labels: List[str]     # N text labels (one per prompt)


class GroundedSAM:
    """Text-prompted instance segmentation.

    Prefers SAM3 direct text→mask (demo.py style) when available.
    Falls back to Grounding-DINO + SAM2 otherwise.
    """

    def __init__(
        self,
        dino_model: str = "IDEA-Research/grounding-dino-base",
        sam_model: str = "facebook/sam2-hiera-large",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.sam3_available = False
        self.dino_available = False
        self._sam_mode = None

        # --- SAM3 (primary: text → mask, no DINO needed) ---
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.sam3_model = build_sam3_image_model()
            self.sam3_processor = Sam3Processor(
                self.sam3_model, confidence_threshold=0.01, device=str(self.device)
            )
            self.sam3_available = True
            self._sam_mode = "sam3"
            print("[GroundedSAM] SAM3 loaded — using direct text-to-mask (no DINO).")
        except Exception as e:
            print(f"[GroundedSAM] SAM3 unavailable ({e}).")

        # --- Grounding-DINO (fallback detection) ---
        if not self.sam3_available:
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

                self.dino_processor = AutoProcessor.from_pretrained(dino_model)
                self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                    dino_model
                ).to(self.device)
                self.dino.eval()
                self.dino_available = True
            except Exception as e:
                print(f"[GroundedSAM] DINO unavailable ({e}).")

            # --- SAM2 (fallback mask refinement) ---
            try:
                from transformers import SamModel, SamProcessor

                self.sam2_processor = SamProcessor.from_pretrained(sam_model)
                self.sam2 = SamModel.from_pretrained(sam_model).to(self.device)
                self.sam2.eval()
                self._sam_mode = "sam2"
            except Exception as e:
                print(f"[GroundedSAM] SAM2 unavailable ({e}).")

    # ── public API ──────────────────────────────────────────────────

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        expand_mask_iters: int = 3,
        expand_kernel: int = 15,
    ) -> SegmentationResult:
        """Segment objects described by text_prompt.

        text_prompt uses '.' as separator: "red mug. black stand."
        Each sub-prompt produces one mask (best-scoring detection).
        """
        if self.sam3_available:
            masks, boxes, scores, labels = self._segment_sam3_direct(image, text_prompt)
        elif self.dino_available:
            masks, boxes, scores, labels = self._segment_dino_fallback(image, text_prompt)
        else:
            return SegmentationResult(
                masks=torch.zeros(0, image.height, image.width, dtype=torch.bool),
                boxes=torch.empty(0, 4),
                scores=torch.empty(0),
                labels=[],
            )

        # Expand masks to recover occluded regions
        if expand_mask_iters > 0 and masks.shape[0] > 0:
            from .utils import dilate_mask

            expanded = []
            for m in masks:
                expanded.append(dilate_mask(m, expand_kernel, expand_mask_iters))
            masks = torch.stack(expanded)

        return SegmentationResult(
            masks=masks, boxes=boxes, scores=scores, labels=labels
        )

    # ── SAM3 direct text→mask (primary) ─────────────────────────────

    @torch.no_grad()
    def _segment_sam3_direct(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """demo.py style: one set_text_prompt call per sub-prompt."""
        W, H = image.size

        # Split "red mug. black stand." → ["red mug", "black stand"]
        prompts = [p.strip() for p in text_prompt.split(".") if p.strip()]

        # SAM3 fused MLP outputs bfloat16 internally; the following nn.Linear
        # needs autocast to upcast/downcast properly. (sam3/examples/*.ipynb)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        all_masks = []
        all_boxes = []
        all_scores = []
        all_labels = []

        with torch.autocast(device_type, dtype=torch.bfloat16):
            # Extract image features once
            clean_state = self.sam3_processor.set_image(image)

            for prompt in prompts:
                # Copy state so each prompt starts clean (demo.py pattern)
                state = clean_state.copy()
                state["backbone_out"] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in clean_state["backbone_out"].items()
                }

                result = self.sam3_processor.set_text_prompt(state=state, prompt=prompt)
                scores = result["scores"]
                masks = result["masks"]
                boxes = result.get("boxes")

                if len(scores) == 0:
                    continue

                best_idx = scores.argmax().item()
                if scores[best_idx].item() < self.box_threshold:
                    continue

                # Resize mask from model resolution → original image size
                mask = masks[best_idx]
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H_m, W_m)
                elif mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                mask = torch.nn.functional.interpolate(
                    mask.float(), size=(H, W), mode="bilinear"
                )[0, 0]

                all_masks.append(mask > 0.5)
                all_scores.append(scores[best_idx])
                all_labels.append(prompt)

                # Convert SAM3 box (cxcywh normalized) → xyxy pixel
                if boxes is not None and len(boxes) > best_idx:
                    b = boxes[best_idx].tolist()
                    cx, cy, bw, bh = b[0], b[1], b[2], b[3]
                    x1 = (cx - bw / 2) * W
                    y1 = (cy - bh / 2) * H
                    x2 = (cx + bw / 2) * W
                    y2 = (cy + bh / 2) * H
                    all_boxes.append([x1, y1, x2, y2])
                else:
                    # No box → derive from mask
                    ys, xs = torch.where(mask > 0.5)
                    if len(ys) > 0:
                        all_boxes.append(
                            [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()]
                        )
                    else:
                        all_boxes.append([0, 0, W, H])

        if not all_masks:
            return (
                torch.zeros(0, H, W, dtype=torch.bool, device=self.device),
                torch.empty(0, 4, device=self.device),
                torch.empty(0, device=self.device),
                [],
            )

        return (
            torch.stack(all_masks).to(self.device),
            torch.tensor(all_boxes, device=self.device),
            torch.tensor([s.item() for s in all_scores], device=self.device),
            all_labels,
        )

    # ── DINO + SAM2 fallback ────────────────────────────────────────

    @torch.no_grad()
    def _segment_dino_fallback(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Grounding-DINO detection + optional SAM2 mask refinement."""
        # Detection
        inputs = self.dino_processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.dino(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"]
        labels = results.get("labels", [text_prompt] * len(boxes))
        scores = results.get("scores", torch.ones(len(boxes)))

        if boxes.shape[0] == 0:
            return (
                torch.zeros(0, image.height, image.width, dtype=torch.bool, device=self.device),
                boxes, scores, list(labels),
            )

        # SAM2 refinement
        if self._sam_mode == "sam2":
            masks = self._refine_sam2(image, boxes)
        else:
            W, H = image.size
            masks = torch.zeros(len(boxes), H, W, dtype=torch.bool, device=self.device)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.int().tolist()
                masks[i, y1:y2, x1:x2] = True

        return masks, boxes, scores, list(labels)

    @torch.no_grad()
    def _refine_sam2(self, image: Image.Image, boxes: torch.Tensor) -> torch.Tensor:
        """SAM2 mask refinement via HuggingFace."""
        inputs = self.sam2_processor(
            image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)
        outputs = self.sam2(**inputs)
        masks = self.sam2_processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=[[image.height, image.width]],
            reshaped_input_sizes=[[image.height, image.width]],
        )[0]
        return masks[:, 0] > 0.5
