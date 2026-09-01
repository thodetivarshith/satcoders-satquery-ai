"""
utils.py — reusable helpers shared across scripts.
"""

import cv2
import time
import threading


class ThreadedCamera:
    """
    Reads frames from the webcam on a background thread so that frame
    capture never blocks the main thread's detection/recognition work.
    This alone typically gives a noticeable FPS boost on CPU-only laptops,
    since cv2.VideoCapture.read() is otherwise a blocking I/O call.
    """

    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera index / permissions.")

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


class FPSCounter:
    """Simple rolling FPS counter for on-screen display."""

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._prev_time = time.time()

    def tick(self):
        now = time.time()
        instant_fps = 1.0 / max(now - self._prev_time, 1e-6)
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        self._prev_time = now
        return self.fps


def draw_label(frame, box, name, confidence, color):
    """Draw a bounding box + name/confidence label, adapted from face_recognition's
    (top, right, bottom, left) box format used throughout this project."""
    top, right, bottom, left = box

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    label = f"{name} ({confidence:.0%})" if name != "Unknown" else "Unknown"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(frame, (left, bottom), (left + text_w + 10, bottom + text_h + 15), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 5, bottom + text_h + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
