"""
gunicorn settings for a multi-process deployment (Linux/macOS):

    cd Code && gunicorn -c gunicorn.conf.py wsgi:app

Each worker process loads its own models and MediaPipe detectors, so
CPU-bound inference runs in parallel instead of contending for one
Python interpreter lock.
"""

import multiprocessing
import os

bind = os.environ.get("ASL_BIND", "127.0.0.1:8000")
workers = int(os.environ.get("ASL_WORKERS", min(4, multiprocessing.cpu_count())))
threads = int(os.environ.get("ASL_THREADS", 4))
worker_class = "gthread"
# Load the app inside each worker: MediaPipe graphs don't survive fork().
preload_app = False
timeout = 30
keepalive = 5
