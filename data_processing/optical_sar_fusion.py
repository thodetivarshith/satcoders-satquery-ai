# Optical-SAR fusion - Kunchala
import numpy as np
def fuse_optical_sar(optical_image, sar_image):
    """
    combine optical and SAR images
    """
    if optical_image.shape[0] != 4:
        raise ValueError("Optical image must have 4 bands.")
    if sar_image.shape[0] != 2:
        raise ValueError("SAR image must have 2 bands.")
    if optical_image.shape[1] != sar_image.shape[1] or  optical_image.shape[2] != sar_image.shape[2]:
      raise ValueError("Optical and SAR images must have the same spatial dimensions.")
    fused_image = np.concatenate((optical_image, sar_image), axis=0)
    return fused_image
def convert_to_hwc(image):
    """Convert image from
    [C, H, W] to [H, W, C]
    """
    return np.transpose(image, (1, 2, 0))

  

