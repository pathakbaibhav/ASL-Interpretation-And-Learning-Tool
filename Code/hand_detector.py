"""
Hand landmark detection with MediaPipe's HandLandmarker (Tasks API), plus
adaptive lighting correction for dark or washed-out frames.

A HandLandmarker instance must not be shared between threads, so the Flask
API keeps one detector per worker thread (see ``api.py``).
"""

import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Pairs of landmark indices joined when drawing a hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Frames darker than this (mean brightness, 0-255) get a second detection
# attempt after lighting correction if the first attempt finds no hand.
DARK_THRESHOLD = 80

_download_lock = threading.Lock()


def ensure_landmarker_model(path=LANDMARKER_PATH):
    """Downloads MediaPipe's hand landmarker model the first time it's needed."""
    path = Path(path)
    with _download_lock:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading hand landmarker model to {path} ...")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(LANDMARKER_URL, tmp)
            tmp.rename(path)
    return path


def frame_brightness(bgr):
    """Returns the mean brightness (0-255) of a BGR frame."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def enhance_lighting(bgr, brightness=None):
    """
    Brightens a dark BGR frame with a gamma curve chosen so its mean maps to
    roughly mid-grey. Gamma alone beat CLAHE + gamma in our low-light tests
    (see README), since CLAHE also amplifies sensor noise.
    """
    if brightness is None:
        brightness = frame_brightness(bgr)
    mean = np.clip(brightness / 255.0, 0.02, 0.98)
    gamma = np.clip(np.log(0.5) / np.log(mean), 0.3, 3.0)
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(bgr, table)


class HandDetection:
    """One detected hand: (21, 2) normalized landmarks plus metadata."""

    def __init__(self, points, handedness, score, lighting_corrected):
        self.points = points
        self.handedness = handedness
        self.score = score
        self.lighting_corrected = lighting_corrected

    def to_dict(self):
        return {
            "landmarks": self.points.round(5).tolist(),
            "handedness": self.handedness,
            "score": round(self.score, 4),
            "lighting_corrected": self.lighting_corrected,
        }


class HandDetector:
    """
    Wraps MediaPipe's HandLandmarker.

    ``video=True`` enables MediaPipe's tracking between frames (faster and
    steadier for webcam loops); ``video=False`` treats every frame on its
    own, which is what the stateless REST API needs.
    """

    def __init__(self, video=False, lighting="auto", min_confidence=0.3,
                 model_path=LANDMARKER_PATH):
        if lighting not in ("auto", "off"):
            raise ValueError("lighting must be 'auto' or 'off'")
        self.video = video
        self.lighting = lighting
        self._last_timestamp = -1

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_landmarker_model(model_path))
            ),
            running_mode=RunningMode.VIDEO if video else RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect(self, bgr, timestamp_ms=None):
        """
        Finds a hand in a BGR frame. Returns a HandDetection, or None if no
        hand is visible. ``timestamp_ms`` is required in video mode.
        """
        result = self._run(bgr, timestamp_ms)
        corrected = False

        # Low-light fallback: only when the raw frame fails *and* is dark,
        # so well-lit frames never pay for a second pass.
        if not result.hand_landmarks and self.lighting == "auto":
            brightness = frame_brightness(bgr)
            if brightness < DARK_THRESHOLD:
                result = self._run(enhance_lighting(bgr, brightness), timestamp_ms)
                corrected = bool(result.hand_landmarks)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        category = result.handedness[0][0] if result.handedness else None
        return HandDetection(
            points=points,
            handedness=category.category_name if category else None,
            score=float(category.score) if category else 0.0,
            lighting_corrected=corrected,
        )

    def _run(self, bgr, timestamp_ms):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if not self.video:
            return self._landmarker.detect(image)

        if timestamp_ms is None:
            raise ValueError("timestamp_ms is required in video mode")
        # MediaPipe rejects timestamps that don't strictly increase, which
        # also covers the retry pass on the same frame.
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp + 1)
        self._last_timestamp = timestamp_ms
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_hand(bgr, points, color=(0, 255, 0), point_color=(0, 0, 255)):
    """Draws a hand skeleton onto a BGR frame in place."""
    h, w = bgr.shape[:2]
    pixels = [(int(x * w), int(y * h)) for x, y in points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, pixels[a], pixels[b], color, 2, cv2.LINE_AA)
    for p in pixels:
        cv2.circle(bgr, p, 3, point_color, -1, cv2.LINE_AA)
    return bgr
