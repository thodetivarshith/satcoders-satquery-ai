"""
GeoChat inference module for SatQuery AI.

Varshith - AI/ML Lead
SIH 2026 | SIH26167

Reusable query_image() interface for the backend.

The GeoChat base model and fine-tuned adapter are NOT stored in GitHub.
Set GEOCHAT_REPO_PATH and GEOCHAT_ADAPTER_PATH in the runtime environment.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("BNB_CUDA_VERSION", "122")

import torch
from PIL import Image


MODEL_NAME = os.environ.get(
    "GEOCHAT_MODEL_NAME",
    "geochat-7B-lora",
)

BASE_MODEL = os.environ.get(
    "GEOCHAT_BASE_MODEL",
    "MBZUAI/geochat-7B",
)

GEOCHAT_REPO_PATH = os.environ.get(
    "GEOCHAT_REPO_PATH",
    "/kaggle/working/GeoChat-main",
)

GEOCHAT_ADAPTER_PATH = os.environ.get(
    "GEOCHAT_ADAPTER_PATH",
    "/root/geochat_checkpoints/geochat_v1_bigearth",
)

MAX_NEW_TOKENS = int(
    os.environ.get("GEOCHAT_MAX_NEW_TOKENS", "24")
)

MODEL_ID = "geochat_v1_bigearth"


_tokenizer = None
_model = None
_image_processor = None
_context_len = None


def _setup_geochat_import() -> None:
    """Add the external GeoChat source to sys.path."""

    if not os.path.isdir(GEOCHAT_REPO_PATH):
        raise FileNotFoundError(
            f"GeoChat source not found: {GEOCHAT_REPO_PATH}. "
            "Set GEOCHAT_REPO_PATH to the GeoChat source directory."
        )

    if GEOCHAT_REPO_PATH not in sys.path:
        sys.path.insert(0, GEOCHAT_REPO_PATH)


def _configure_image_processor(processor):
    """Force the 504x504 CLIP input used by the fine-tuned checkpoint."""

    processor.size = {
        "shortest_edge": 504
    }

    processor.crop_size = {
        "height": 504,
        "width": 504
    }

    return processor


def _get_input_device(model):
    """Return the device used by the language-model input embeddings."""

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


def load_model(force_reload: bool = False):
    """Load the fine-tuned GeoChat adapter once and reuse it."""

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

    if not os.path.isdir(GEOCHAT_ADAPTER_PATH):
        raise FileNotFoundError(
            f"Fine-tuned GeoChat adapter not found: "
            f"{GEOCHAT_ADAPTER_PATH}. "
            "Set GEOCHAT_ADAPTER_PATH to the trained adapter directory."
        )

    from geochat.model.builder import load_pretrained_model

    print("Loading fine-tuned GeoChat...")
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {GEOCHAT_ADAPTER_PATH}")
    print(f"GeoChat source: {GEOCHAT_REPO_PATH}")

    (
        tokenizer,
        model,
        image_processor,
        context_len,
    ) = load_pretrained_model(
        GEOCHAT_ADAPTER_PATH,
        BASE_MODEL,
        MODEL_NAME,
        load_8bit=False,
        load_4bit=True,
        device_map={"": "cuda:0"},
        device="cuda",
    )

    image_processor = _configure_image_processor(
        image_processor
    )

    _tokenizer = tokenizer
    _model = model
    _image_processor = image_processor
    _context_len = context_len

    print("Fine-tuned GeoChat loaded successfully.")
    print("Image processor: 504x504")

    return (
        _tokenizer,
        _model,
        _image_processor,
        _context_len,
    )


def _max_tokens_for_query(query_text: str) -> int:
    """Use shorter decoding for common closed-form questions."""

    q = query_text.lower().strip()

    if q.startswith(
        (
            "would you",
            "is ",
            "are ",
            "does ",
            "do ",
            "can ",
            "has ",
            "have ",
        )
    ):
        return min(MAX_NEW_TOKENS, 16)

    if any(
        x in q
        for x in (
            "choose",
            "option",
            "which of the following",
            "letter",
        )
    ):
        return min(MAX_NEW_TOKENS, 16)

    return MAX_NEW_TOKENS


def query_image(
    image_path: str,
    query_text: str,
) -> Dict[str, Any]:
    """
    Ask the fine-tuned GeoChat model a question about an RGB satellite image.

    Returns:
        {
            "answer": str,
            "confidence": float,
            "model": str,
            "processing_time_ms": int
        }

    Note:
        confidence is currently a placeholder (0.0), not a calibrated score.
    """

    if not image_path:
        raise ValueError(
            "image_path must not be empty."
        )

    if not query_text or not query_text.strip():
        raise ValueError(
            "query_text must not be empty."
        )

    image_file = Path(image_path)

    if not image_file.is_file():
        raise FileNotFoundError(
            f"Satellite image not found: {image_path}"
        )

    (
        tokenizer,
        model,
        image_processor,
        _,
    ) = load_model()

    from geochat.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )

    from geochat.conversation import conv_templates

    from geochat.mm_utils import (
        process_images,
        tokenizer_image_token,
    )

    image = Image.open(
        image_file
    ).convert("RGB")

    start_time = time.perf_counter()

    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    vision_device = model.get_vision_tower().device

    if isinstance(image_tensor, list):

        image_tensor = [
            x.to(
                device=vision_device,
                dtype=torch.float16,
            )
            for x in image_tensor
        ]

    else:

        image_tensor = image_tensor.to(
            device=vision_device,
            dtype=torch.float16,
        )

    prompt = (
        DEFAULT_IMAGE_TOKEN
        + "\n"
        + query_text.strip()
    )

    conv = conv_templates["v1"].copy()

    conv.append_message(
        conv.roles[0],
        prompt,
    )

    conv.append_message(
        conv.roles[1],
        None,
    )

    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)

    input_device = _get_input_device(model)

    input_ids = input_ids.to(
        input_device
    )

    max_tokens = _max_tokens_for_query(
        query_text
    )

    with torch.inference_mode():

        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_tokens,
            use_cache=True,
        )

    generated_ids = output_ids[
        :,
        input_ids.shape[1]:
    ]

    generated_ids = generated_ids[
        generated_ids != IMAGE_TOKEN_INDEX
    ]

    if generated_ids.numel() > 0:

        answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    else:

        answer = ""

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    processing_time_ms = int(
        (
            time.perf_counter()
            - start_time
        ) * 1000
    )

    return {
        "answer": answer,
        "confidence": 0.0,
        "model": MODEL_ID,
        "processing_time_ms": processing_time_ms,
    }


def unload_model() -> None:
    """Release GeoChat and GPU memory."""

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

    print(
        "GeoChat inference module"
    )

    print(
        f"Base model: {BASE_MODEL}"
    )

    print(
        f"Adapter: {GEOCHAT_ADAPTER_PATH}"
    )

    print(
        f"Max new tokens: {MAX_NEW_TOKENS}"
    )

    print(
        "Deterministic decoding: enabled"
    )
