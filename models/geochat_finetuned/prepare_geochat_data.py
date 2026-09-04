# Data converter for official GeoChat fine-tuning - Varshith
"""
prepare_geochat_data.py

GeoChat's official train_mem.py (forked from LLaVA) does its own data
loading — it does NOT use bigearth_dataset.py. It expects:

  1. A flat folder of plain RGB image files (.jpg/.png)
  2. A single JSON file, LLaVA/GeoChat "conversation" format:

     [
       {
         "id": "sample_001",
         "image": "sample_001.jpg",   # filename inside the image folder
         "conversations": [
           {"from": "human", "value": "<image>\nQuestion text"},
           {"from": "gpt", "value": "Answer text"}
         ]
       },
       ...
     ]

This script converts our BigEarthNet-style patches (or the flat SAR-only
sample_XX.tif files from the current test batch) into that format:
  - Extracts/reads the optical RGB bands, saves as .jpg
  - Writes conversations.json alongside them

Usage:
    python prepare_geochat_data.py \
        --input_dir /path/to/bigearthnet/patches \
        --annotations_file /path/to/annotations.jsonl \
        --output_image_dir /path/to/geochat_data/images \
        --output_json /path/to/geochat_data/conversations.json
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from PIL import Image  # type: ignore[reportMissingImports]
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")


S2_RGB_BANDS = ["B04", "B03", "B02"]  # red, green, blue
S2_REFLECTANCE_MAX = 10000.0


def _read_band(path: str) -> np.ndarray:
    if rasterio is not None and os.path.exists(path):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    npy_path = os.path.splitext(path)[0] + ".npy"
    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.float32)
    raise FileNotFoundError(f"Could not find band file: {path} (or .npy fallback)")


def patch_to_rgb_uint8(patch_dir: str, patch_id: str) -> np.ndarray:
    """Reads R/G/B Sentinel-2 bands for one patch, returns HWC uint8 for saving as .jpg."""
    bands = []
    for band in S2_RGB_BANDS:
        path = os.path.join(patch_dir, f"{patch_id}_{band}.tif")
        arr = _read_band(path)
        arr = np.clip(arr / S2_REFLECTANCE_MAX, 0.0, 1.0)
        bands.append(arr)
    rgb = np.stack(bands, axis=-1)
    return (rgb * 255).astype(np.uint8)


def convert(
    input_dir: str,
    annotations_file: str,
    output_image_dir: str,
    output_json: str,
) -> None:
    os.makedirs(output_image_dir, exist_ok=True)

    with open(annotations_file, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    conversations: List[Dict] = []
    skipped = 0

    for sample in samples:
        patch_id = sample["patch_id"]
        patch_dir = os.path.join(input_dir, patch_id)

        try:
            rgb_uint8 = patch_to_rgb_uint8(patch_dir, patch_id)
        except FileNotFoundError as e:
            print(f"Skipping {patch_id}: {e}")
            skipped += 1
            continue

        image_filename = f"{patch_id}.jpg"
        Image.fromarray(rgb_uint8).save(os.path.join(output_image_dir, image_filename))

        conversations.append({
            "id": patch_id,
            "image": image_filename,
            "conversations": [
                {"from": "human", "value": f"<image>\n{sample['question']}"},
                {"from": "gpt", "value": sample["answer"]},
            ],
        })

    with open(output_json, "w") as f:
        json.dump(conversations, f, indent=2)

    print(f"Converted {len(conversations)} samples ({skipped} skipped). "
          f"Images: {output_image_dir}, JSON: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="BigEarthNet patches root")
    parser.add_argument("--annotations_file", required=True, help="Our annotations.jsonl")
    parser.add_argument("--output_image_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    convert(args.input_dir, args.annotations_file, args.output_image_dir, args.output_json)