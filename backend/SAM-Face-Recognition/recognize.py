"""
recognize.py — Step 2: real-time face detection + recognition.

Pipeline per frame:
  1. Downscale frame for speed, detect faces (HOG or CNN model).
  2. Compute a 128-d embedding for each detected face.
  3. Compare against the enrolled database using Euclidean distance in
     embedding space; the closest match under MATCH_TOLERANCE wins.
  4. Draw bounding boxes + name + confidence, overlay FPS.

Performance techniques used (this is what makes it "advanced" vs a naive loop):
  - Threaded camera capture (utils.ThreadedCamera) to avoid I/O blocking.
  - Frame downscaling before detection (config.PROCESS_SCALE).
  - Detection only every Nth frame (config.DETECT_EVERY_N_FRAMES); boxes are
    reused on skipped frames so the video still looks smooth.

Usage:
    python recognize.py
"""

import os
import cv2
import pickle
import numpy as np
import face_recognition

import config
from utils import ThreadedCamera, FPSCounter, draw_label


def load_database():
    if not os.path.exists(config.ENCODINGS_FILE):
        raise FileNotFoundError(
            f"No encodings database found at {config.ENCODINGS_FILE}. "
            "Run enroll.py first to add at least one known person."
        )
    with open(config.ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def flatten_database(db):
    """Turn {name: [enc, enc, ...]} into parallel arrays for fast vectorized matching."""
    names, encodings = [], []
    for name, enc_list in db.items():
        for enc in enc_list:
            names.append(name)
            encodings.append(enc)
    return names, np.array(encodings)


def match_face(face_encoding, known_names, known_encodings):
    if len(known_encodings) == 0:
        return "Unknown", 0.0

    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]

    if best_distance <= config.MATCH_TOLERANCE:
        confidence = 1.0 - best_distance  # rough confidence score for display
        return known_names[best_idx], confidence
    return "Unknown", 1.0 - best_distance


def recognize():
    db = load_database()
    known_names, known_encodings = flatten_database(db)
    print(f"Loaded {len(known_names)} embeddings for {len(set(known_names))} known people.")

    cam = ThreadedCamera(config.CAM_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    fps_counter = FPSCounter()

    frame_count = 0
    last_results = []  # cached (box, name, confidence) between detection frames

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            run_detection = (frame_count % config.DETECT_EVERY_N_FRAMES == 0)

            if run_detection:
                small = cv2.resize(frame, (0, 0), fx=config.PROCESS_SCALE, fy=config.PROCESS_SCALE)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                boxes_small = face_recognition.face_locations(rgb_small, model=config.DETECTION_MODEL)
                encodings = face_recognition.face_encodings(rgb_small, boxes_small)

                last_results = []
                scale = 1.0 / config.PROCESS_SCALE
                for box, enc in zip(boxes_small, encodings):
                    top, right, bottom, left = [int(v * scale) for v in box]
                    name, confidence = match_face(enc, known_names, known_encodings)
                    last_results.append(((top, right, bottom, left), name, confidence))

            for box, name, confidence in last_results:
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
                draw_label(frame, box, name, confidence, color)

            fps = fps_counter.tick()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Face Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
