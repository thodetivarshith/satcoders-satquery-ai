# Preprocessing - Kunchala

import numpy as np


def normalize_band(band):
    """
    Normalize a single band of a GeoTIFF image to the range [0, 1].
    """
    band = band.astype(np.float32)

    min_val = np.min(band)
    max_val = np.max(band)

    if max_val - min_val == 0:
        return np.zeros_like(band)

    return (band - min_val) / (max_val - min_val)


def preprocess_optical(optical_image):
    """
    Select and normalize 4 optical bands.
    Bands: B2, B3, B4, B8 (Blue, Green, Red, NIR)
    """

    if optical_image.shape[0] < 4:
        raise ValueError("Optical image must have at least 4 bands.")

    optical_image = optical_image[:4]

    processed = []

    for band in optical_image:
        processed.append(normalize_band(band))

    return np.stack(processed, axis=0)


def preprocess_sar(sar_image):
    """
    Select and normalize 2 SAR bands.
    Bands: VV, VH
    """

    if sar_image.shape[0] < 2:
        raise ValueError("SAR image must have at least 2 bands.")

    sar = sar_image[:2]

    VV = normalize_band(sar[0])
    VH = normalize_band(sar[1])

    return np.stack([VV, VH], axis=0)