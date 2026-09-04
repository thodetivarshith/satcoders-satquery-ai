import os
import re
import json
import time

import planetary_computer
import rasterio
import numpy as np
import pandas as pd

from PIL import Image
from pystac_client import Client
from rasterio.windows import Window


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\thode\OneDrive\satcoders-satquery-ai"

MANIFEST = os.path.join(
    ROOT,
    "data_processing",
    "BigEarthNet",
    "geochat_500_image_manifest.csv"
)

ANNOTATIONS = os.path.join(
    ROOT,
    "data_processing",
    "BigEarthNet",
    "geochat_500_annotations.csv"
)

BNE_TEXT = os.path.join(
    ROOT,
    "data_processing",
    "BigEarthNet",
    "BigEarthNet.txt.parquet"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "models",
    "geochat_finetuned",
    "images"
)

CONVERSATIONS = os.path.join(
    ROOT,
    "models",
    "geochat_finetuned",
    "conversations.json"
)

FAILED_FILE = os.path.join(
    ROOT,
    "data_processing",
    "BigEarthNet",
    "failed_patches.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

manifest = pd.read_csv(MANIFEST)
annotations = pd.read_csv(ANNOTATIONS)

print(f"Manifest patches: {len(manifest)}")
print(f"Annotations: {len(annotations)}")

print("Loading BigEarthNet metadata...")

bne = pd.read_parquet(
    BNE_TEXT,
    columns=["patch_id", "latitude", "longitude"]
)

# One coordinate row per patch.
bne = bne.drop_duplicates("patch_id")

patches = manifest.merge(
    bne,
    on="patch_id",
    how="left"
)

missing_coords = patches["latitude"].isna().sum()

if missing_coords:
    raise RuntimeError(
        f"{missing_coords} patches have no latitude/longitude."
    )

print("All 499 patches have coordinates.")


# ============================================================
# PLANETARY COMPUTER
# ============================================================

STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

stac_client = Client.open(STAC_URL)

scene_cache = {}


# ============================================================
# PARSE BIGEARTHNET PATCH ID
# ============================================================

def parse_patch_id(patch_id):

    parts = patch_id.rsplit("_", 2)

    if len(parts) != 3:
        raise ValueError(
            f"Invalid patch ID: {patch_id}"
        )

    scene_name = parts[0]
    x_index = int(parts[1])
    y_index = int(parts[2])

    match = re.match(
        r"^(S2[AB])_MSIL2A_(\d{8}T\d{6})_N\d+_R\d+_(T\d{2}[A-Z]{3})$",
        scene_name
    )

    if match is None:
        raise ValueError(
            f"Cannot parse Sentinel-2 scene: {patch_id}"
        )

    platform = match.group(1)
    acquisition = match.group(2)
    tile = match.group(3)

    return (
        scene_name,
        platform,
        acquisition,
        tile,
        x_index,
        y_index
    )


# ============================================================
# FIND EXACT STAC SCENE
# ============================================================

def find_scene(
    patch_id,
    latitude,
    longitude
):

    (
        scene_name,
        platform,
        acquisition,
        tile,
        x_index,
        y_index
    ) = parse_patch_id(patch_id)

    # Scene cache avoids repeating STAC queries
    # for patches from the same Sentinel-2 scene.
    if scene_name in scene_cache:
        return (
            scene_cache[scene_name],
            x_index,
            y_index
        )

    date = acquisition[:8]

    start = (
        f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        "T00:00:00Z"
    )

    end = (
        f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        "T23:59:59Z"
    )

    # Small spatial search around the BigEarthNet patch.
    search = stac_client.search(
        collections=["sentinel-2-l2a"],
        bbox=[
            longitude - 0.05,
            latitude - 0.05,
            longitude + 0.05,
            latitude + 0.05
        ],
        datetime=f"{start}/{end}",
        max_items=50
    )

    items = list(search.items())

    # Exact tile + exact acquisition.
    candidates = [
        item
        for item in items
        if tile in item.id
        and acquisition in item.id
        and item.id.startswith(platform)
    ]

    if not candidates:

        raise RuntimeError(
            f"No exact STAC scene for {patch_id}. "
            f"Spatial results: {len(items)}"
        )

    scene = candidates[0]

    scene_cache[scene_name] = scene

    print(
        f"Scene found: {scene.id}"
    )

    return (
        scene,
        x_index,
        y_index
    )


# ============================================================
# READ RGB PATCH
# ============================================================

def read_rgb_patch(
    scene,
    x_index,
    y_index
):

    window = Window(
        x_index * 120,
        y_index * 120,
        120,
        120
    )

    arrays = []

    # RGB:
    # B04 = Red
    # B03 = Green
    # B02 = Blue

    for band in ["B04", "B03", "B02"]:

        href = planetary_computer.sign(
            scene.assets[band].href
        )

        with rasterio.open(href) as ds:

            arr = ds.read(
                1,
                window=window
            ).astype(np.float32)

        arr = np.clip(
            arr / 3000.0,
            0,
            1
        )

        arrays.append(
            (arr * 255).astype(np.uint8)
        )

    return np.dstack(arrays)


# ============================================================
# DOWNLOAD / CREATE 499 RGB IMAGES
# ============================================================

success = 0
failed = []

for index, row in patches.iterrows():

    patch_id = row["patch_id"]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{patch_id}.jpg"
    )

    try:

        if os.path.exists(output_file):

            success += 1

            print(
                f"[{index + 1}/{len(patches)}] "
                f"Already exists: {patch_id}"
            )

            continue

        scene, x_index, y_index = find_scene(
            patch_id,
            float(row["latitude"]),
            float(row["longitude"])
        )

        rgb = read_rgb_patch(
            scene,
            x_index,
            y_index
        )

        Image.fromarray(rgb).save(
            output_file,
            quality=95
        )

        success += 1

        print(
            f"[{index + 1}/{len(patches)}] "
            f"Saved: {patch_id}.jpg"
        )

    except Exception as e:

        failed.append({
            "patch_id": patch_id,
            "error": str(e)
        })

        print(
            f"[{index + 1}/{len(patches)}] "
            f"FAILED: {patch_id}"
        )

        print(
            "Error:",
            e
        )

        # Small pause to avoid hammering the API.
        time.sleep(1)


# ============================================================
# SAVE FAILURE REPORT
# ============================================================

if failed:

    pd.DataFrame(failed).to_csv(
        FAILED_FILE,
        index=False
    )

    print(
        f"Failed patch report: {FAILED_FILE}"
    )


# ============================================================
# CREATE GE0CHAT CONVERSATIONS.JSON
# ONLY FOR AVAILABLE IMAGES
# ============================================================

records = []

available_images = {
    filename[:-4]
    for filename in os.listdir(OUTPUT_DIR)
    if filename.lower().endswith(".jpg")
}

for _, row in annotations.iterrows():

    patch_id = row["patch_id"]

    # Never create a training record whose image is missing.
    if patch_id not in available_images:
        continue

    records.append({
        "id": patch_id,
        "image": f"{patch_id}.jpg",
        "conversations": [
            {
                "from": "human",
                "value": (
                  "<image>\n"
                      + str(row["input"])
                )
            },
            {
                "from": "gpt",
                "value": str(row["output"])
            }
        ]
    })


with open(
    CONVERSATIONS,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"Images created: {success}")
print(f"Failed images:  {len(failed)}")
print(f"GeoChat records: {len(records)}")

print()
print("Images:")
print(OUTPUT_DIR)

print()
print("Conversations:")
print(CONVERSATIONS)

if failed:
    print()
    print(
        "Some patches failed. "
        "See failed_patches.csv."
    )