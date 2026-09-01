"""
enroll.py — Step 1: enroll a known person.

Captures several webcam samples, detects the face in each, and computes a
128-dimensional face embedding (dlib ResNet, via face_recognition). These
embeddings — not raw pixels — are what gets compared at recognition time,
which is what makes this approach robust to lighting/angle changes.

Usage:
    python enroll.py --name Hasini
"""

import os
import pickle
import argparse
import sys
from builtins import ImportError, open, ord, print, str

try:
    import cv2
except ImportError:
    print("Missing dependency: OpenCV is required. Install with 'pip install opencv-python'.")
    sys.exit(1)

try:
    import face_recognition
except ImportError:
    print("Missing dependency: face_recognition is required. Install with 'pip install face_recognition'.")
    sys.exit(1)

import config
from utils import ThreadedCamera


def load_database():
    if os.path.exists(config.ENCODINGS_FILE):
        with open(config.ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_database(db):
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump(db, f)


def enroll(name: str):
    db = load_database()
    db.setdefault(name, [])

    cam = ThreadedCamera(config.CAM_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    print(f"Enrolling '{name}'. Capturing {config.SAMPLES_PER_PERSON} good samples.")
    print("Move your head slightly between captures (angle/lighting variety helps accuracy).")
    print("Press 'q' to stop early.")

    collected = 0
    try:
        while collected < config.SAMPLES_PER_PERSON:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model=config.DETECTION_MODEL)

            display = frame.copy()
            if len(boxes) == 1:
                encodings = face_recognition.face_encodings(rgb, boxes)
                db[name].append(encodings[0])
                collected += 1

                top, right, bottom, left = boxes[0]
                cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(display, f"Captured {collected}/{config.SAMPLES_PER_PERSON}",
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif len(boxes) > 1:
                cv2.putText(display, "Multiple faces detected - only one person at a time",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Enrollment - press q to stop", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    save_database(db)
    print(f"Done. '{name}' now has {len(db[name])} stored embeddings in {config.ENCODINGS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the person to enroll")
    args = parser.parse_args()
    enroll(args.name)