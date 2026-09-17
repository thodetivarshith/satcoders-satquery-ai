# Preprocessing - Kunchala
import numpy as np


def normalize_band(band):
    """
    Normalize a single image band to the range [0, 1].
    Output data type: float32
    """
    band = band.astype(np.float32)

    min_val = np.min(band)
    max_val = np.max(band)

    # Avoid division by zero for a constant band
    if max_val - min_val == 0:
        return np.zeros_like(band, dtype=np.float32)

    return (band - min_val) / (max_val - min_val)


def preprocess_optical(optical_image):
    """
    Preprocess Sentinel-2 optical imagery.

    Expected bands:
    B2 - Blue
    B3 - Green
    B4 - Red
    B8 - NIR

    Input shape:
        (bands, height, width)

    Output shape:
        (4, height, width)

    Output values:
        0 to 1
    """

    if optical_image.ndim != 3:
        raise ValueError(
            "Optical image must have shape (bands, height, width)."
        )

    if optical_image.shape[0] < 4:
        raise ValueError(
            "Optical image must have at least 4 bands."
        )

    # Select first 4 bands
    optical = optical_image[:4]

    processed = []

    for band in optical:
        processed.append(normalize_band(band))

    return np.stack(processed, axis=0).astype(np.float32)


def preprocess_sar(sar_image):
    """
    Preprocess Sentinel-1 SAR imagery.

    Expected channels:
    VV
    VH

    Input shape:
        (2, height, width)

    Output shape:
        (2, height, width)

    Output values:
        0 to 1
    """

    if sar_image.ndim != 3:
        raise ValueError(
            "SAR image must have shape (bands, height, width)."
        )

    if sar_image.shape[0] < 2:
        raise ValueError(
            "SAR image must have at least 2 bands: VV and VH."
        )

    # Select the first two SAR channels
    sar = sar_image[:2]

    # First channel: VV
    VV = normalize_band(sar[0])

    # Second channel: VH
    VH = normalize_band(sar[1])

    return np.stack([VV, VH], axis=0).astype(np.float32)