"""
WSGI entry point for running the API under gunicorn with several worker
processes (see gunicorn.conf.py). Settings come from environment variables:

  ASL_BACKEND       ensemble | rf | mlp            (default ensemble)
  ASL_CACHE_SIZE    response cache entries, 0=off  (default 4096)
  ASL_FRAME_SLOTS   concurrent frame detections per worker (default 4)
  ASL_BATCHING      1 to micro-batch predictions, 0 to disable (default 1)
"""

import os

from api import create_app
from predictor import Predictor

app = create_app(
    predictor=Predictor(backend=os.environ.get("ASL_BACKEND", "ensemble")),
    cache_size=int(os.environ.get("ASL_CACHE_SIZE", 4096)),
    max_concurrent_frames=int(os.environ.get("ASL_FRAME_SLOTS", 4)),
    warm_detectors=True,
    batching=os.environ.get("ASL_BATCHING", "1") == "1",
)
