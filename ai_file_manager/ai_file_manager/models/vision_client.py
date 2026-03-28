"""
models/vision_client.py — Qwen2-VL 2B image description via HuggingFace Transformers.
Uses 4-bit quantization (bitsandbytes) to fit on 4 GB VRAM.
"""

from __future__ import annotations
import base64
import io
import threading
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger

log = get_logger("vision_client")

IMAGE_PROMPT = (
    "Describe this image in detail. Include: main subject, setting, "
    "objects present, text visible, colors, and any identifying information. "
    "Be specific and factual."
)


class VisionClient:
    """
    Singleton wrapper for Qwen2-VL-2B-Instruct.
    Falls back gracefully if GPU memory is insufficient.
    """

    _instance: Optional["VisionClient"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cfg = load_config().models.vision
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "VisionClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ─── Model loading ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "transformers / torch not installed. "
                "Run: pip install transformers torch torchvision"
            )

        model_id = self._cfg.model_id
        use_4bit = bool(self._cfg.load_in_4bit)
        target_device = str(self._cfg.device)

        log.info(f"Loading vision model: {model_id}")

        kwargs: dict = {
            "torch_dtype": torch.float16 if target_device == "cuda" else torch.float32,
            "device_map": "auto" if target_device == "cuda" else "cpu",
        }

        if use_4bit and target_device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["quantization_config"] = bnb_config
                kwargs.pop("torch_dtype", None)
                log.info("4-bit quantization enabled for vision model")
            except ImportError:
                log.warning("bitsandbytes not found — loading without 4-bit quant")

        try:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, **kwargs
            )
            self._processor = AutoProcessor.from_pretrained(model_id)

            if target_device == "cuda" and not kwargs.get("quantization_config"):
                self._model = self._model.to("cuda")

            self._device = target_device
            log.info(f"Vision model loaded ✓ (device={self._device})")

        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                log.warning("VRAM insufficient — falling back to CPU for vision model")
                kwargs = {"device_map": "cpu", "torch_dtype": torch.float32}
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **kwargs
                )
                self._processor = AutoProcessor.from_pretrained(model_id)
                self._device = "cpu"
            else:
                raise

    # ─── Inference ────────────────────────────────────────────────────────────

    def describe_image(
        self,
        image_path: Path,
        max_size_px: int = 1024,
    ) -> Optional[str]:
        """
        Return a text description of the image at image_path.
        Resizes image before sending to limit VRAM usage.
        Returns None on failure.
        """
        try:
            from PIL import Image
        except ImportError:
            log.error("Pillow not installed: pip install Pillow")
            return None

        try:
            img = Image.open(image_path).convert("RGB")
            img = _resize_image(img, max_size_px)
        except Exception as exc:
            log.error(f"Cannot open image {image_path.name}: {exc}")
            return None

        with self._model_lock:
            self._load_model()

            try:
                import torch

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": IMAGE_PROMPT},
                        ],
                    }
                ]

                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=[text],
                    images=[img],
                    padding=True,
                    return_tensors="pt",
                )

                if self._device == "cuda":
                    inputs = inputs.to("cuda")

                with torch.no_grad():
                    generated_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=int(self._cfg.max_new_tokens),
                        do_sample=False,
                    )

                # Strip input tokens from output
                input_len = inputs["input_ids"].shape[1]
                out_ids = generated_ids[:, input_len:]
                result = self._processor.batch_decode(
                    out_ids, skip_special_tokens=True
                )[0]

                log.debug(f"Vision output for {image_path.name!r}: {result[:100]}")
                return result.strip()

            except Exception as exc:
                log.error(f"Vision inference error for {image_path.name}: {exc}")
                return None

    def describe_image_pil(self, pil_image) -> Optional[str]:
        """Describe a PIL Image object directly (used by video pipeline)."""
        with self._model_lock:
            self._load_model()
            try:
                import torch

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": IMAGE_PROMPT},
                        ],
                    }
                ]
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=[text], images=[pil_image], padding=True, return_tensors="pt"
                )
                if self._device == "cuda":
                    inputs = inputs.to("cuda")

                with torch.no_grad():
                    gen = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)

                out = gen[:, inputs["input_ids"].shape[1]:]
                return self._processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            except Exception as exc:
                log.error(f"Vision PIL inference error: {exc}")
                return None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resize_image(img, max_px: int = 1024):
    """Resize image so the longest side ≤ max_px, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))
