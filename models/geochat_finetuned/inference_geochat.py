
import os
import time
from pathlib import Path

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
os.environ.setdefault("BNB_CUDA_VERSION", "122")
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")

BASE_MODEL = "MBZUAI/geochat-7B"

# Fine-tuned adapter is optional
GEOCHAT_ADAPTER_PATH = os.environ.get(
    "GEOCHAT_ADAPTER_PATH",
    "/kaggle/working/satcoders-satquery-ai/checkpoints/geochat_v1_bigearth"
)

MODEL_NAME = "geochat-7B-lora"
GEOCHAT_REPO_PATH = "/kaggle/working/GeoChat-main"

_tokenizer = None
_model = None
_image_processor = None
_context_len = None
_model_type = None


def _setup_geochat_import():
    import sys

    if GEOCHAT_REPO_PATH not in sys.path:
        sys.path.insert(0, GEOCHAT_REPO_PATH)


def _configure_image_processor(image_processor):
    if hasattr(image_processor, "size"):
        image_processor.size = {"shortest_edge": 504}

    if hasattr(image_processor, "crop_size"):
        image_processor.crop_size = {"height": 504, "width": 504}

    return image_processor


def load_model(force_reload=False):
    global _tokenizer
    global _model
    global _image_processor
    global _context_len
    global _model_type

    if _model is not None and not force_reload:
        return (
            _tokenizer,
            _model,
            _image_processor,
            _context_len,
        )

    _setup_geochat_import()

    from geochat.model.builder import load_pretrained_model
    from geochat.mm_utils import get_model_name_from_path

    adapter_exists = (
        bool(GEOCHAT_ADAPTER_PATH)
        and Path(GEOCHAT_ADAPTER_PATH).is_dir()
        and (
            Path(GEOCHAT_ADAPTER_PATH, "adapter_model.bin").exists()
            or Path(GEOCHAT_ADAPTER_PATH, "adapter_model.safetensors").exists()
        )
    )

    if adapter_exists:
        print("Loading fine-tuned GeoChat...")
        print("Adapter:", GEOCHAT_ADAPTER_PATH)

        model_path = GEOCHAT_ADAPTER_PATH
        model_name = MODEL_NAME
        _model_type = "geochat_v1_bigearth"
    else:
        print("Fine-tuned adapter unavailable.")
        print("Loading pretrained GeoChat...")

        model_path = BASE_MODEL
        model_name = get_model_name_from_path(BASE_MODEL)
        _model_type = "geochat-7B-pretrained"

    tokenizer, model, image_processor, context_len = (
        load_pretrained_model(
            model_path,
            BASE_MODEL if adapter_exists else None,
            model_name,
            load_8bit=False,
            load_4bit=True,
            device_map={"": "cuda:0"},
            device="cuda",
        )
    )

    image_processor = _configure_image_processor(
        image_processor
    )

    _tokenizer = tokenizer
    _model = model
    _image_processor = image_processor
    _context_len = context_len

    print("✅ GeoChat loaded:", _model_type)

    return (
        _tokenizer,
        _model,
        _image_processor,
        _context_len,
    )


def _max_tokens(query):
    q = query.lower()

    if any(x in q for x in [
        "yes or no",
        "does the image",
        "is there",
        "are there",
        "whether",
    ]):
        return 16

    if any(x in q for x in [
        "where",
        "location",
        "coordinate",
        "coordinates",
        "bbox",
        "bounding box",
    ]):
        return 64

    return 48


def query_image(image_path, query_text):
    if not image_path:
        raise ValueError("image_path is required")

    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")

    from PIL import Image

    (
        tokenizer,
        model,
        image_processor,
        _,
    ) = load_model()

    from geochat.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
    )

    from geochat.mm_utils import (
        process_images,
        tokenizer_image_token,
    )

    from geochat.conversation import conv_templates

    image_path = str(image_path)
    query_text = query_text.strip()

    image = Image.open(image_path).convert("RGB")

    # GeoChat visual encoder resolution
    image = image.resize((504, 504))

    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]

    image_tensor = image_tensor.to(
        device="cuda",
        dtype=model.dtype,
    )

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    conv = conv_templates["v1"].copy()

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + query_text

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to("cuda")

    max_tokens = _max_tokens(query_text)

    start = time.perf_counter()

    with __import__("torch").inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_tokens,
            use_cache=True,
        )

    processing_time_ms = (
        time.perf_counter() - start
    ) * 1000

    generated_ids = output_ids[:, input_ids.shape[1]:]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    # Simple demo confidence.
    # This is NOT a calibrated probability.
    confidence = 0.80 if answer else 0.0

    return {
        "answer": answer,
        "confidence": confidence,
        "model": _model_type,
        "processing_time_ms": round(processing_time_ms, 2),
    }
