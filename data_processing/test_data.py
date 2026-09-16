# Data tests - Kunchala
import numpy as np
from preprocess import preprocess_optical, preprocess_sar
from optical_sar_fusion import fuse_optical_sar
from optical_sar_fusion import convert_to_hwc

#create simple Sentinel-2 and Sentinel-2 optical data
optical_data = np.random.rand(4, 256, 256)

#Create simple Sentinel-1 SAR data
sar_data = np.random.rand(2, 256, 256)
# preprocess_optical data
optical_processed = preprocess_optical(optical_data)
# preprocess_SAR data
sar_processed = preprocess_sar(sar_data)
 #fuse optical and SAR data
fused_image = fuse_optical_sar(optical_processed, sar_processed)

#Convert fused image to HWC format
final_image_hwc = convert_to_hwc(fused_image)
#display results
print("Optical Processed Shape:", optical_processed.shape)
print("SAR Processed Shape:", sar_processed.shape)
print("Fused Image Shape:", fused_image.shape)
print("Final Image HWC Shape:", final_image_hwc.shape)



