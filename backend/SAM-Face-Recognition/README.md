# Task 5 — Advanced Face Detection & Recognition
**SAM AI Technologies internship submission**

## What this is
A real-time face recognition system using deep-learning face embeddings
(dlib's 128-d ResNet model, via the `face_recognition` library) instead of
classical methods like Haar Cascade + LBPH. This is the same family of
approach used in production systems (FaceNet/ArcFace-style embedding +
distance matching).

## Why this counts as "advanced"
| Aspect | Basic version | This version |
|---|---|---|
| Detection | Haar Cascade | HOG/CNN deep detector (dlib) |
| Recognition | LBPH (pixel patterns) | 128-d deep embeddings + distance metric |
| Matching | Fixed classifier retrain | Vectorized nearest-neighbor matching, tunable tolerance |
| Performance | Blocking webcam reads | Threaded capture + frame-skipping + downscaling |
| Architecture | Single script | Modular: config / utils / enroll / recognize |

## Setup
```bash
pip install -r requirements.txt
```
No compiler/CMake needed — `dlib-bin` ships prebuilt wheels for Windows/Mac/Linux.

## Usage
**1. Enroll people** (run once per person):
```bash
python enroll.py --name Varshith
python enroll.py --name Friend
```
Look slightly left/right/up/down between captures for better accuracy across angles.

**2. Run live recognition:**
```bash
python recognize.py
```
Press `q` to quit. Known faces get a green box + name + confidence.
Unrecognized faces get a red box labeled "Unknown".

## Tuning
All knobs live in `config.py`:
- `MATCH_TOLERANCE` — lower = stricter (fewer false matches, more "Unknown")
- `DETECTION_MODEL` — `"hog"` (fast, CPU) or `"cnn"` (accurate, needs GPU)
- `DETECT_EVERY_N_FRAMES` / `PROCESS_SCALE` — trade accuracy for FPS

## Files
- `config.py` — all settings in one place
- `utils.py` — threaded webcam capture, FPS counter, drawing helpers
- `enroll.py` — captures samples for a new known person
- `recognize.py` — live detection + recognition loop
- `encodings.pkl` — generated after enrollment (your face database)

## Notes for the report
Mention in your submission that this uses a **deep embedding-based approach**
(dlib ResNet-34, trained on ~3M faces) rather than classical Haar+LBPH, and
that performance optimizations (threading, frame-skipping, downscaling) were
added deliberately for real-time CPU inference — both are good talking
points for demonstrating you understand the engineering trade-offs, not
just calling a library function.
