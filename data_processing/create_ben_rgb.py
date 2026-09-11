from pathlib import Path

import cv2
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "BigEarthNet-S2-selected"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_processing"
    / "data"
    / "bigearthnet_rgb"
)


def normalize_band(band):
    """Convert a Sentinel-2 uint16 band to uint8."""

    band = band.astype(np.float32)

    low = np.percentile(
        band,
        2
    )

    high = np.percentile(
        band,
        98
    )

    if high <= low:
        return np.zeros_like(
            band,
            dtype=np.uint8
        )

    band = (
        band - low
    ) / (
        high - low
    )

    band = np.clip(
        band,
        0.0,
        1.0
    )

    return (
        band * 255
    ).astype(np.uint8)


def find_patch_folder():
    """Find the first folder that actually contains B02/B03/B04."""

    for patch_dir in sorted(
        DATASET_ROOT.rglob("*")
    ):

        if not patch_dir.is_dir():
            continue

        patch_name = patch_dir.name

        b02 = patch_dir / f"{patch_name}_B02.tif"
        b03 = patch_dir / f"{patch_name}_B03.tif"
        b04 = patch_dir / f"{patch_name}_B04.tif"

        if (
            b02.exists()
            and b03.exists()
            and b04.exists()
        ):
            return patch_dir

    return None


def main():
    """Create an RGB image from B04/B03/B02."""

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}"
        )

    patch_dir = find_patch_folder()

    if patch_dir is None:
        raise FileNotFoundError(
            "Could not find a complete S2 patch "
            "containing B02, B03 and B04."
        )

    patch_name = patch_dir.name

    print(
        f"Using patch: {patch_name}"
    )

    red_path = (
        patch_dir
        / f"{patch_name}_B04.tif"
    )

    green_path = (
        patch_dir
        / f"{patch_name}_B03.tif"
    )

    blue_path = (
        patch_dir
        / f"{patch_name}_B02.tif"
    )

    print(
        "Red:  ",
        red_path
    )

    print(
        "Green:",
        green_path
    )

    print(
        "Blue: ",
        blue_path
    )

    # ---------------------------------------------
    # Read Sentinel-2 bands
    # ---------------------------------------------

    with rasterio.open(
        red_path
    ) as src:
        red = src.read(1)

    with rasterio.open(
        green_path
    ) as src:
        green = src.read(1)

    with rasterio.open(
        blue_path
    ) as src:
        blue = src.read(1)

    print(
        "Band shapes:"
    )

    print(
        "B04:",
        red.shape
    )

    print(
        "B03:",
        green.shape
    )

    print(
        "B02:",
        blue.shape
    )

    # ---------------------------------------------
    # Normalize bands
    # ---------------------------------------------

    red = normalize_band(red)
    green = normalize_band(green)
    blue = normalize_band(blue)

    # ---------------------------------------------
    # Build RGB
    # ---------------------------------------------

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=2
    )

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{patch_name}_RGB.png"
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )
    )

    print()
    print(
        "RGB image created successfully:"
    )

    print(
        output_path
    )

    print()
    print(
        "RGB shape:",
        rgb.shape
    )


if __name__ == "__main__":
    main()
    