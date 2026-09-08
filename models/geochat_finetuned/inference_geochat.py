"""
GeoChat inference module for SatQuery AI.

Varshith - AI/ML Lead
SIH 2026 | SIH26167

Provides a reusable query_image() interface for the backend.

The GeoChat base model is NOT stored in this repository.
Set GEOCHAT_REPO_PATH to the local GeoChat source directory when needed.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image


MODEL_NAME = os.environ.get(
    "GEOCHAT_MODEL_NAME",
    "MBZUAI/geochat-7B",
)

GEOCHAT_REPO_PATH = os.environ.get(
    "GEOCHAT_REPO_PATH",
    "/kaggle/working/GeoChat-main",
)

MAX_NEW_TOKENS = int(
    os.environ.get("GEOCHAT_MAX_NEW_TOKENS", "24")
)

TEMPERATURE = float(
    os.environ.get("GEOCHAT_TEMPERATURE", "0.2")
)

MODEL_ID = "geochat_v1_bigearth"


_tokenizer = None
_model = None
_image_processor = None
_context_len = None


def _setup_geochat_import():
    """Add the external GeoChat source to sys.path."""

    if not os.path.isdir(GEOCHAT_REPO_PATH):
        raise FileNotFoundError(
            f"GeoChat source not found: {GEOCHAT_REPO_PATH}\n"
            "Set GEOCHAT_REPO_PATH to the directory containing "
            "the GeoChat package."
        )

    if GEOCHAT_REPO_PATH not in sys.path:
        sys.path.insert(0, GEOCHAT_REPO_PATH)


def _configure_image_processor(processor):
    """
    Configure CLIP for GeoChat's 504x504 visual input.

    The GeoChat checkpoint used here expects 1297 CLIP positional
    embeddings = 36x36 patches + CLS token.
    """

    processor.size = {"shortest_edge": 504}
    processor.crop_size = {
        "height": 504,
        "width": 504,
    }

    return processor


def load_model(force_reload: bool = False):
    """
    Load GeoChat once and reuse it for subsequent queries.

    Uses 4-bit quantization to reduce VRAM usage on T4 GPUs.
    """

    global _tokenizer
    global _model
    global _image_processor
    global _context_len

    if _model is not None and not force_reload:
        return (
            _tokenizer,
            _model,
            _image_processor,
            _context_len,
        )

    _setup_geochat_import()

    from geochat.model.builder import load_pretrained_model

    print("Loading GeoChat...")
    print(f"Model: {MODEL_NAME}")
    print(f"GeoChat source: {GEOCHAT_REPO_PATH}")

    tokenizer, model, image_processor, context_len = (
        load_pretrained_model(
            MODEL_NAME,
            None,
            "geochat",
            load_4bit=True,
            device_map="auto",
        )
    )

    image_processor = _configure_image_processor(
        image_processor
    )

    _tokenizer = tokenizer
    _model = model
    _image_processor = image_processor
    _context_len = context_len

    print("GeoChat loaded successfully.")
    print("Image processor: 504x504")

    return (
        _tokenizer,
        _model,
        _image_processor,
        _context_len,
    )


def _get_input_device(model):
    """Find the device used by the model input embeddings."""

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def query_image(
    image_path: str,
    query_text: str,
) -> Dict[str, Any]:
    """
    Ask GeoChat a natural-language question about a satellite image.

    Returns:
        {
            "answer": str,
            "confidence": float,
            "model": str,
            "processing_time_ms": int
        }
    """

    if not image_path:
        raise ValueError("image_path must not be empty.")

    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty.")

    image_file = Path(image_path)

    if not image_file.is_file():
        raise FileNotFoundError(
            f"Satellite image not found: {image_path}"
        )

    (
        tokenizer,
        model,
        image_processor,
        context_len,
    ) = load_model()

    image = Image.open(image_file).convert("RGB")

    start_time = time.perf_counter()

    image_tensor = image_processor.preprocess(
        image,
        return_tensors="pt",
    )["pixel_values"]

    # Move image to the vision tower's device.
    vision_device = None

    try:
        vision_device = model.get_vision_tower().device
    except Exception:
        vision_device = _get_input_device(model)

    image_tensor = image_tensor.to(
        device=vision_device,
        dtype=torch.float16,
    )

    # GeoChat uses the LLaVA conversation format.
    from geochat.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from geochat.conversation import conv_templates
    from geochat.mm_utils import tokenizer_image_token

    prompt = (
        DEFAULT_IMAGE_TOKEN
        + "\n"
        + query_text.strip()
    )

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)

    input_device = _get_input_device(model)
    input_ids = input_ids.to(input_device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    output_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    processing_time_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    return {
        "answer": output_text,
        "confidence": 0.0,
        "model": MODEL_ID,
        "processing_time_ms": processing_time_ms,
    }


def unload_model():
    """Release the loaded GeoChat model and GPU memory."""

    global _tokenizer
    global _model
    global _image_processor
    global _context_len

    _tokenizer = None
    _model = None
    _image_processor = None
    _context_len = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("GeoChat inference module")
    print(f"Model: {MODEL_NAME}")
    print(f"GeoChat source: {GEOCHAT_REPO_PATH}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
