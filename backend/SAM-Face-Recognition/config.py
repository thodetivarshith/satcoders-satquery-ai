"""
config.py — central configuration for the face recognition system.
Keeping every tunable in one place is what makes this "modular" rather
than a single monolithic script.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")          # raw captured face images, per person
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pkl")  # trained face embeddings database

# --- Capture settings ---
SAMPLES_PER_PERSON = 25
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Detection model ---
# "hog"  -> CPU, fast, good for real-time on a laptop (default)
# "cnn"  -> much more accurate, needs GPU/CUDA or will be slow on CPU
DETECTION_MODEL = "hog"

# --- Recognition ---
# Lower = stricter matching (fewer false positives, more false "Unknown")
# Typical range: 0.4 (strict) - 0.6 (lenient). 0.5 is a solid default.
MATCH_TOLERANCE = 0.5

# Resize frame before processing to speed up detection on CPU (advanced perf trick)
PROCESS_SCALE = 0.5  # process at 50% resolution, then scale boxes back up

# Skip N-1 out of every N frames for detection to boost FPS, while still
# drawing boxes every frame (tracked boxes persist between detections)
DETECT_EVERY_N_FRAMES = 3
