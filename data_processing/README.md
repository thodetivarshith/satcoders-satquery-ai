# SatQuery AI – Remote Sensing Data Processing

## Overview

The `data_processing` module prepares satellite imagery for the
SatQuery AI system.

It processes Sentinel-2 Optical and Sentinel-1 SAR data, performs
normalization and validation, and combines the data into a
6-channel representation for downstream AI and backend processing.

---

## 1. Processing Pipeline

The complete processing flow is:

```text
Sentinel-2 Optical + Sentinel-1 SAR
                ↓
           Load GeoTIFF
                ↓
      Preprocess Optical
          B2 / B3 / B4 / B8
                ↓
      Preprocess SAR
             VV / VH
                ↓
      Validate Dimensions
        and Input Data
                ↓
      Optical + SAR Fusion
                ↓
       6-Channel Output
                ↓
        Backend / AI Model
        2. Sentinel-2 Optical Preprocessing

The optical preprocessing pipeline uses four Sentinel-2 bands:

Band	Description
B2	Blue
B3	Green
B4	Red
B8	Near Infrared (NIR)
Expected Input
(bands, height, width)

At least four optical channels are required.

Expected Channel Order

The preprocessing pipeline expects the source data to provide the
bands in this order:

Channel 0 → B2
Channel 1 → B3
Channel 2 → B4
Channel 3 → B8

The GeoTIFF loader preserves the original source band order.

Therefore, the input data must provide the bands in the expected order.

Output
(4, height, width)

The four optical channels are independently normalized to the range:

0 to 1

Output data type:

float32
3. Sentinel-1 SAR Preprocessing

The SAR preprocessing pipeline uses two Sentinel-1 SAR channels:

Channel	Description
VV	Vertical transmit / Vertical receive
VH	Vertical transmit / Horizontal receive
Expected Input
(2, height, width)
Expected Channel Order

The preprocessing pipeline expects:

Channel 0 → VV
Channel 1 → VH

The GeoTIFF loader preserves the original source band order.

Therefore, the source data must provide the SAR channels in the
expected VV/VH order.

Output
(2, height, width)

Each SAR channel is independently normalized to:

0 to 1

Output data type:

float32
4. GeoTIFF Loading

The geotiff_loader.py module loads satellite imagery using
Rasterio.

The loader reads:

Image pixel data
Image width
Image height
Number of bands
Coordinate Reference System (CRS)
Geospatial transform

The loader preserves the original band order supplied by the
GeoTIFF source.

5. Data Normalization

Each input band is converted to float32 and normalized using
Min-Max normalization.

The formula is:

normalized_value =
(value - minimum) / (maximum - minimum)

After normalization, the values are within:

[0, 1]

If a band contains a constant value, division by zero is avoided
and a zero-valued output is returned.

6. Input Validation

The preprocessing pipeline validates the input data before fusion.

The following conditions are checked:

Invalid input dimensions
Insufficient optical bands
Insufficient SAR channels
Empty or invalid inputs
NaN values
Infinite values
Mismatched optical and SAR spatial dimensions

Invalid inputs generate appropriate validation errors.

7. Optical + SAR Fusion

After preprocessing:

Optical:
(4, H, W)

SAR:
(2, H, W)

The two datasets are concatenated along the channel dimension.

Therefore:

4 Optical Channels
        +
2 SAR Channels
        =
6 Fused Channels

The final fused output is:

(6, H, W)

For channels-last applications, the output can be converted to:

(H, W, 6)
8. Tested Output

The pipeline was tested with the following dimensions:

Optical Shape : (4, 256, 256)
SAR Shape     : (2, 256, 256)
Fused Shape   : (6, 256, 256)
HWC Shape     : (256, 256, 6)

This confirms that four optical channels and two SAR channels are
successfully combined into a six-channel representation.

9. Performance Testing

The preprocessing pipeline was tested for processing performance.

Test result:

Average Processing Time: approximately 1.8 ms
Performance Requirement: < 500 ms
Result: PASSED

Multiple sample inputs were also tested to verify consistent
processing performance.

10. Sample SAR Data

Sample SAR files used for testing:

data/optical/sar/sample_sar_1.tif
data/optical/sar/sample_sar_2.tif

Each sample TIFF contains one raster channel.

The preprocessing pipeline uses the two channels as:

Channel 0 → VV
Channel 1 → VH

The sample TIFF metadata does not explicitly contain polarization
labels. Therefore, the VV/VH assignment is based on the expected
source channel order and should be confirmed from the original
Sentinel-1 data source when available.

11. Output Specification
Data	Channels	Example Shape	Value Range	Data Type
Sentinel-2 Optical	4	(4, 256, 256)	0–1	float32
Sentinel-1 SAR	2	(2, 256, 256)	0–1	float32
Fused Output	6	(6, 256, 256)	0–1	float32
HWC Output	6	(256, 256, 6)	0–1	float32
12. Module Structure
data_processing/
│
├── geotiff_loader.py
├── preprocess.py
├── optical_sar_fusion.py
├── test_data.py
├── README.md
│
└── data/
    └── optical/
        └── sar/
            ├── sample_sar_1.tif
            └── sample_sar_2.tif
File Responsibilities
geotiff_loader.py

Loads GeoTIFF satellite imagery and extracts image and metadata.

preprocess.py

Performs optical and SAR preprocessing and normalization.

optical_sar_fusion.py

Validates dimensions and combines four optical channels with two SAR
channels to produce a six-channel output.

test_data.py

Tests preprocessing, validation, fusion, and processing performance.

README.md

Documents the remote sensing preprocessing and fusion pipeline.

13. Final Data Flow
              Sentinel-2 Optical
                       │
             ┌─────────┴─────────┐
             │                   │
          B2 Blue            B3 Green
          B4 Red             B8 NIR
             │                   │
             └─────────┬─────────┘
                       ↓
                Normalization
                       ↓
                 4 Channels
                       │
                       │
                       ├──────────────┐
                       │              │
                       │              │
                       │       Sentinel-1 SAR
                       │              │
                       │         ┌────┴────┐
                       │         │         │
                       │        VV        VH
                       │         │         │
                       │         └────┬────┘
                       │              ↓
                       │       Normalization
                       │              ↓
                       │         2 Channels
                       │              │
                       └──────┬───────┘
                              ↓
                    Optical + SAR Fusion
                              ↓
                       6-Channel Output
                              ↓
                       (6, H, W)
                              ↓
                      HWC Conversion
                              ↓
                       (H, W, 6)
                              ↓
                      Backend / AI Model
14. Role in SatQuery AI

The data_processing module provides standardized satellite data
for downstream SatQuery AI components.

The main responsibilities are:

Load satellite imagery from GeoTIFF files.
Validate input dimensions and data.
Preprocess Sentinel-2 optical bands.
Preprocess Sentinel-1 SAR channels.
Normalize the input bands.
Fuse optical and SAR information.
Produce a standardized six-channel representation.
Provide validated data for downstream backend and AI components.

The resulting six-channel representation can be passed to
downstream computer vision, multimodal fusion, and AI analysis
components.

15. Completion Status

The remote sensing preprocessing pipeline has been implemented and
tested for:

GeoTIFF loading
Optical preprocessing
SAR preprocessing
Data normalization
Input validation
Optical + SAR fusion
Six-channel output
HWC conversion
Performance testing
Documentation

The pipeline is structured for downstream backend and AI model
integration.