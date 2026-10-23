"""
Locations of model files, and the download of MediaPipe's hand landmarker.

Kept free of heavy imports so build scripts can use it without loading
MediaPipe or OpenCV.
"""

import os
import tempfile
import threading
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

_download_lock = threading.Lock()


def ensure_landmarker_model(path=LANDMARKER_PATH):
    """
    Downloads MediaPipe's hand landmarker model the first time it's needed.

    Several processes (e.g. gunicorn workers) may start at once, so each
    downloads into its own temporary file and atomically moves it into
    place; whichever finishes last wins, and readers never see a partial file.
    """
    path = Path(path)
    with _download_lock:
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {path} ...")
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(LANDMARKER_URL, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return path
