# BigEarthNet data loader - Varshith
"""
bigearth_dataset.py

PyTorch Dataset for fine-tuning GeoChat on BigEarthNet.txt-style data:
optical (Sentinel-2) + SAR (Sentinel-1) fused into the team's shared
6-band tensor, paired with VQA-style question/answer text.

TEAM DATA CONTRACT ("Format 1: Image Input" — agreed Sep 9 integration
meeting, used by Kunchala, Abhinay, Akshaya, and this loader):

    {
        "data": np.ndarray([H, W, 6], dtype=float32),  # normalized [0, 1]
        "bands": ["red", "green", "blue", "nir", "vv", "vh"],
        "shape": [H, W],
        "crs": "EPSG:4326",
        "spatial_resolution_m": 10.0,
    }

This loader produces that exact 6-band tensor per sample so it's a
drop-in match for Kunchala's `load_and_preprocess_geotiff()` output —
if her `data_processing/preprocess.py` is ready by the time you train,
swap `_load_and_fuse` below to just call her function directly.

IMPORTANT — channel mismatch with GeoChat itself:
GeoChat's pretrained vision encoder (CLIP-based ViT) is 3-channel RGB.
It cannot natively consume a 6-band tensor. So every sample carries
BOTH:
  - "fused_bands": the full [6, H, W] tensor (team contract, used
    later if you add a channel-adapter or hand off to Abhinay/Akshaya)
  - "pixel_values": a [3, H, W] RGB slice, ImageNet-normalized, that
    GeoChat's encoder actually trains/infers on right now

This keeps you compatible with the team format without blocking on
building a 6-channel encoder adapter under time pressure. If you want
true optical-SAR fusion INSIDE GeoChat later (not just at the data
layer), that's a config.py / model-surgery decision — flag it to me
when you get there, it's a non-trivial change (replacing the encoder's
first conv layer and re-initializing weights for the extra channels).

Expected on-disk layout (adjust ROOT_DIR / manifest path in config.py):

    bigearthnet_root/
        patches/
            <patch_id>/
                <patch_id>_B04.tif   # Sentinel-2 red
                <patch_id>_B03.tif   # Sentinel-2 green
                <patch_id>_B02.tif   # Sentinel-2 blue
                <patch_id>_B08.tif   # Sentinel-2 NIR
                <patch_id>_VV.tif    # Sentinel-1 (optional per patch)
                <patch_id>_VH.tif
        annotations.jsonl           # one JSON object per line, from
                                     # BigEarthNet.txt (optical+SAR+text):
            {"patch_id": "...", "question": "...", "answer": "...",
             "has_sar": true}

If Kunchala's actual pipeline emits a different on-disk format, only
`_load_optical` / `_load_sar` / `_load_annotations` need to change —
the rest (fusion, prompt formatting, batching) stays the same either way.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
from torch.utils.data import Dataset  # type: ignore[import-not-found]

try:
    import rasterio  # type: ignore[import-not-found]
except ImportError:
    rasterio = None  # fall back to .npy if rasterio isn't installed


# Sentinel-2 bands used in the team's 4-optical-band contract
S2_OPTICAL_BANDS = ["B04", "B03", "B02", "B08"]  # red, green, blue, nir
BAND_NAMES = ["red", "green", "blue", "nir", "vv", "vh"]  # team contract order

# Sentinel-2 reflectance values are typically 0-10000 (uint16-ish)
S2_REFLECTANCE_MAX = 10000.0
# Sentinel-1 SAR backscatter is in dB, roughly -25 to 0
SAR_DB_MIN, SAR_DB_MAX = -25.0, 0.0

# GeoChat / most CLIP-style vision encoders expect this
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BigEarthDataset(Dataset):
    """
    Loads one (image, text) training example at a time for GeoChat
    instruction fine-tuning.

    Each item is returned as a dict:
        {
            "patch_id": str,
            "pixel_values": FloatTensor [3, H, W]   # optical RGB, normalized
            "sar_values": FloatTensor [2, H, W] or None,  # VV, VH if present
            "prompt": str,      # formatted instruction for GeoChat
            "answer": str,      # target text
        }
    """

    def __init__(
        self,
        root_dir: str,
        annotations_file: str,
        split: str = "train",
        split_ids_file: Optional[str] = None,
        image_size: int = IMAGE_SIZE,
        use_sar: bool = True,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.use_sar = use_sar
        self.split = split

        self.samples = self._load_annotations(annotations_file)

        if split_ids_file is not None and os.path.exists(split_ids_file):
            with open(split_ids_file, "r") as f:
                allowed_ids = {line.strip() for line in f if line.strip()}
            self.samples = [s for s in self.samples if s["patch_id"] in allowed_ids]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples loaded for split='{split}'. Check root_dir="
                f"'{root_dir}', annotations_file='{annotations_file}', "
                f"split_ids_file='{split_ids_file}'."
            )

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #

    def _load_annotations(self, annotations_file: str) -> List[Dict]:
        samples = []
        with open(annotations_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def _band_path(self, patch_id: str, band: str) -> str:
        return os.path.join(self.root_dir, "patches", patch_id, f"{patch_id}_{band}.tif")

    def _read_band(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing band file: {path}")
        if rasterio is not None:
            with rasterio.open(path) as src:
                return src.read(1).astype(np.float32)
        # fallback: assume a co-located .npy with the same stem
        npy_path = os.path.splitext(path)[0] + ".npy"
        if os.path.exists(npy_path):
            return np.load(npy_path).astype(np.float32)
        raise RuntimeError(
            "rasterio is not installed and no .npy fallback was found for "
            f"{path}. Install rasterio (`pip install rasterio`) to read .tif bands."
        )

    def _load_optical(self, patch_id: str) -> np.ndarray:
        """Returns the 4 team-contract optical bands (R,G,B,NIR), HWC, float32 in [0, 1]."""
        bands = []
        for band in S2_OPTICAL_BANDS:
            arr = self._read_band(self._band_path(patch_id, band))
            arr = np.clip(arr / S2_REFLECTANCE_MAX, 0.0, 1.0)
            bands.append(arr)
        optical = np.stack(bands, axis=-1)  # H, W, 4  (red, green, blue, nir)
        return optical

    def _load_sar(self, patch_id: str) -> Optional[np.ndarray]:
        """Returns VV/VH stack, HWC (H, W, 2), float32 normalized to [0, 1], or None."""
        vv_path = self._band_path(patch_id, "VV")
        vh_path = self._band_path(patch_id, "VH")
        if not (os.path.exists(vv_path) and os.path.exists(vh_path)):
            return None
        vv = self._read_band(vv_path)
        vh = self._read_band(vh_path)
        vv = np.clip((vv - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN), 0.0, 1.0)
        vh = np.clip((vh - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN), 0.0, 1.0)
        return np.stack([vv, vh], axis=-1)  # H, W, 2

    def _load_and_fuse(self, patch_id: str) -> Dict:
        """
        Builds the team-contract 6-band tensor: [H, W, 6] = (red, green,
        blue, nir, vv, vh), values in [0, 1].

        If SAR is missing for this patch, vv/vh channels are zero-filled
        and `has_sar=False` is reported — matches the fusion module's
        "handle missing bands" edge case in Kunchala's spec, so the
        shape contract never breaks even for optical-only patches.

        Swap this method's body for a direct call to Kunchala's
        `load_and_preprocess_geotiff()` once `data_processing/preprocess.py`
        is merged — same output shape, so nothing downstream changes.
        """
        optical = self._load_optical(patch_id)  # H, W, 4
        sar = self._load_sar(patch_id)  # H, W, 2 or None
        has_sar = sar is not None
        if sar is None:
            h, w = optical.shape[:2]
            sar = np.zeros((h, w, 2), dtype=np.float32)
        fused = np.concatenate([optical, sar], axis=-1)  # H, W, 6
        return {"data": fused, "has_sar": has_sar}

    # ------------------------------------------------------------------ #
    # Image processing
    # ------------------------------------------------------------------ #

    def _resize(self, img: np.ndarray, size: int) -> np.ndarray:
        """Nearest-neighbor resize without adding a heavy dependency.
        Swap for cv2.resize / PIL if you need bilinear/antialiasing quality."""
        h, w = img.shape[:2]
        if (h, w) == (size, size):
            return img
        row_idx = (np.linspace(0, h - 1, size)).astype(np.int32)
        col_idx = (np.linspace(0, w - 1, size)).astype(np.int32)
        return img[row_idx][:, col_idx]

    def _to_geochat_pixel_values(self, fused_hwc: np.ndarray) -> torch.Tensor:
        """Slices RGB (first 3 of the 6 fused bands) and prepares it the way
        GeoChat's CLIP-based encoder expects: resized, ImageNet-normalized, CHW."""
        rgb_hwc = fused_hwc[:, :, :3]  # red, green, blue
        rgb_hwc = self._resize(rgb_hwc, self.image_size)
        rgb_hwc = (rgb_hwc - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(rgb_hwc, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(chw)

    def _to_fused_tensor(self, fused_hwc: np.ndarray) -> torch.Tensor:
        """Full 6-band tensor at native resolution, team-contract order
        (red, green, blue, nir, vv, vh), CHW, values already in [0, 1]."""
        chw = np.transpose(fused_hwc, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(chw)

    # ------------------------------------------------------------------ #
    # Prompt formatting — matches GeoChat's instruction-tuning format
    # ------------------------------------------------------------------ #

    def _format_prompt(self, question: str, has_sar: bool) -> str:
        prefix = (
            "You are a remote sensing analyst. Analyze the provided "
            "satellite imagery"
        )
        if has_sar:
            prefix += " (optical and SAR) "
        else:
            prefix += " "
        return f"{prefix}and answer the question.\nQuestion: {question}\nAnswer:"

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        patch_id = sample["patch_id"]

        fusion_result = self._load_and_fuse(patch_id)
        fused_hwc = fusion_result["data"]  # H, W, 6
        has_sar = fusion_result["has_sar"] and self.use_sar

        pixel_values = self._to_geochat_pixel_values(fused_hwc)  # 3, 224, 224 — feeds GeoChat now
        fused_bands = self._to_fused_tensor(fused_hwc)  # 6, H, W — team contract, native res

        prompt = self._format_prompt(sample["question"], has_sar)

        return {
            "patch_id": patch_id,
            "pixel_values": pixel_values,
            "fused_bands": fused_bands,
            "has_sar": has_sar,
            "prompt": prompt,
            "answer": sample["answer"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """fused_bands are native-resolution and may vary in H/W across patches,
    so they're kept as a list rather than stacked — resize/crop them
    yourself downstream if you need a batched fusion tensor. pixel_values
    are always IMAGE_SIZE x IMAGE_SIZE so those stack cleanly. Text fields
    are left as lists for the tokenizer step in train_geochat.py."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

    return {
        "patch_id": [b["patch_id"] for b in batch],
        "pixel_values": pixel_values,
        "fused_bands": [b["fused_bands"] for b in batch],
        "has_sar": [b["has_sar"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }


if __name__ == "__main__":
    # Quick smoke test — point these at a real data dir to sanity-check loading.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--annotations_file", required=True)
    args = parser.parse_args()

    ds = BigEarthDataset(root_dir=args.root_dir, annotations_file=args.annotations_file)
    print(f"Loaded {len(ds)} samples.")
    example = ds[0]
    print("patch_id:", example["patch_id"])
    print("pixel_values shape (GeoChat input):", example["pixel_values"].shape)
    print("fused_bands shape (team contract):", example["fused_bands"].shape)
    print("has_sar:", example["has_sar"])
    print("prompt:", example["prompt"])
    print("answer:", example["answer"])
