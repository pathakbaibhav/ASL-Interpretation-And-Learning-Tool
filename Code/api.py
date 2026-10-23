"""
Flask REST API for real-time ASL letter recognition, and the web frontend.

Endpoints
  GET  /                     web app (webcam, live translation, practice mode)
  GET  /api/health           status, available backends and labels
  POST /api/predict          classify one hand from its 21 landmarks
  POST /api/predict/batch    classify up to 256 hands in one call
  POST /api/predict/frame    find the hand in a camera frame, then classify it
  GET  /api/metrics          request counts, latency percentiles, cache stats

Run with:  python api.py [--port 8000] [--threads 8] [--no-cache]
"""

import argparse
import base64
import binascii
import queue
import threading
import time
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, g, jsonify, render_template, request, send_from_directory

from batcher import MicroBatcher
from cache import LRUCache
from features import normalized_features, quantize
from hand_detector import HandDetector, ensure_landmarker_model
from metrics import RequestMetrics
from predictor import Predictor

HERE = Path(__file__).resolve().parent
SIGN_IMAGES_DIR = HERE / "signImages"

MAX_BATCH = 256
MAX_TOP_K = 5
MAX_FRAME_SIDE = 1280  # Larger frames are downscaled before detection.


class DetectorPool:
    """
    A fixed set of MediaPipe detectors shared by the request threads.

    MediaPipe detectors aren't thread-safe, so each request borrows one for
    the duration of a detection. The pool size bounds how many frames are
    processed at once; ``warm=True`` builds every detector up front so no
    request pays the ~300 ms model start-up cost.
    """

    def __init__(self, size, factory, warm=False):
        self._factory = factory
        self._idle = queue.LifoQueue()
        self._lock = threading.Lock()
        self._created = 0
        self._all = []
        self.size = size
        if warm:
            for _ in range(size):
                self._idle.put(self._new())

    def _new(self):
        detector = self._factory()
        self._created += 1
        self._all.append(detector)
        return detector

    def acquire(self, timeout):
        """Returns an idle detector, or None if all are busy for ``timeout`` seconds."""
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self.size:
                return self._new()
        try:
            return self._idle.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self, detector):
        self._idle.put(detector)

    def close(self):
        """Closes every detector. Open MediaPipe graphs stop the process from exiting."""
        with self._lock:
            detectors, self._all = self._all, []
        for detector in detectors:
            close = getattr(detector, "close", None)
            if close:
                close()


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def create_app(predictor=None, cache_size=4096, max_concurrent_frames=8,
               detector_factory=HandDetector, warm_detectors=False, batching=True):
    """
    Builds the Flask app.

    cache_size=0 disables the response cache. batching=True routes single
    predictions through a MicroBatcher so concurrent requests share one
    vectorized model call. max_concurrent_frames bounds how
    many frames run hand detection at once; a request that can't get a
    detector within a second gets HTTP 503 instead of piling up behind the
    others and making every client slow.
    """
    app = Flask(
        __name__,
        template_folder=str(HERE / "web" / "templates"),
        static_folder=str(HERE / "web" / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

    predictor = predictor or Predictor()
    cache = LRUCache(cache_size) if cache_size else None
    metrics = RequestMetrics()
    batcher = MicroBatcher(predictor) if batching else None
    detectors = DetectorPool(max_concurrent_frames, detector_factory, warm=warm_detectors)

    app.extensions["asl"] = {"predictor": predictor, "cache": cache, "metrics": metrics,
                             "detectors": detectors}

    def timed(endpoint):
        """Records latency and adds a Server-Timing header to the response."""
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                g.start = time.perf_counter()
                error = False
                try:
                    response = fn(*args, **kwargs)
                except ApiError as e:
                    error = True
                    response = jsonify({"error": e.message})
                    response.status_code = e.status
                elapsed = (time.perf_counter() - g.start) * 1000
                metrics.record(endpoint, elapsed, error=error)
                response.headers["Server-Timing"] = f"app;dur={elapsed:.2f}"
                return response
            return wrapper
        return decorator

    def elapsed_ms():
        return round((time.perf_counter() - g.start) * 1000, 3)

    def options_from(body):
        backend = body.get("backend") or request.args.get("backend") or predictor.default_backend
        if backend not in predictor.backends:
            raise ApiError(f"Unknown backend {backend!r}; choose from {list(predictor.backends)}")
        try:
            top_k = int(body.get("top_k", request.args.get("top_k", 3)))
        except (TypeError, ValueError):
            raise ApiError("top_k must be an integer")
        return backend, max(1, min(top_k, MAX_TOP_K))

    def json_body():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ApiError("Request body must be a JSON object")
        return body

    def to_features(landmarks):
        try:
            return normalized_features(landmarks)
        except (ValueError, TypeError, KeyError, IndexError) as e:
            raise ApiError(f"Invalid landmarks: {e}")

    def run_model(features, backend, top_k):
        if batcher is not None:
            return batcher.predict(features, backend, top_k)
        return predictor.predict(features, backend, top_k)

    def classify(features, backend, top_k):
        """Classifies one hand, reusing a cached answer for near-identical input."""
        if cache is None:
            return run_model(features, backend, top_k), False
        key = (backend, top_k, quantize(features).tobytes())
        result = cache.get(key)
        if result is not None:
            return result, True
        result = run_model(features, backend, top_k)
        cache.put(key, result)
        return result, False

    def decode_frame():
        """Reads an image from a raw body, a multipart upload or base64 JSON."""
        if "image" in request.files:
            raw = request.files["image"].read()
        elif request.mimetype and request.mimetype.startswith("image/"):
            raw = request.get_data()
        else:
            body = json_body()
            data = body.get("image")
            if not isinstance(data, str):
                raise ApiError("Send an image as the body, a multipart 'image' field, or base64 JSON")
            if data.startswith("data:"):
                data = data.split(",", 1)[-1]
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                raise ApiError("Image is not valid base64")

        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        if frame is None:
            raise ApiError("Could not decode the image")
        h, w = frame.shape[:2]
        if max(h, w) > MAX_FRAME_SIDE:
            scale = MAX_FRAME_SIDE / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "Request body too large (limit 4 MB)"}), 413

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return jsonify({"error": "Method not allowed"}), 405

    @app.get("/")
    def index():
        return render_template("index.html", labels=list(predictor.labels))

    @app.get("/signs/<path:name>")
    def sign_image(name):
        return send_from_directory(SIGN_IMAGES_DIR, name, max_age=86400)

    @app.get("/models/hand_landmarker.task")
    def landmarker_model():
        # Served locally so the browser engine doesn't depend on Google's CDN.
        path = ensure_landmarker_model()
        return send_from_directory(path.parent, path.name, max_age=86400)

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "backends": list(predictor.backends),
            "default_backend": predictor.default_backend,
            "labels": list(predictor.labels),
            "cache_enabled": cache is not None,
        })

    @app.post("/api/predict")
    @timed("predict")
    def predict():
        body = json_body()
        backend, top_k = options_from(body)
        if "landmarks" not in body:
            raise ApiError("Missing 'landmarks': 21 [x, y] pairs from MediaPipe")
        result, cached = classify(to_features(body["landmarks"]), backend, top_k)
        return jsonify({**result, "cached": cached, "latency_ms": elapsed_ms()})

    @app.post("/api/predict/batch")
    @timed("predict_batch")
    def predict_batch():
        body = json_body()
        backend, top_k = options_from(body)
        items = body.get("items")
        if not isinstance(items, list) or not items:
            raise ApiError("'items' must be a non-empty list of landmark sets")
        if len(items) > MAX_BATCH:
            raise ApiError(f"At most {MAX_BATCH} items per batch")
        features = []
        for i, landmarks in enumerate(items):
            try:
                features.append(normalized_features(landmarks))
            except (ValueError, TypeError, KeyError, IndexError) as e:
                raise ApiError(f"Invalid landmarks in item {i}: {e}")
        results = predictor.predict_batch(features, backend, top_k)
        return jsonify({"results": results, "count": len(results), "latency_ms": elapsed_ms()})

    @app.post("/api/predict/frame")
    @timed("predict_frame")
    def predict_frame():
        body = request.get_json(silent=True) if request.is_json else {}
        backend, top_k = options_from(body or {})
        frame = decode_frame()

        detector = detectors.acquire(timeout=1.0)
        if detector is None:
            raise ApiError("Server busy, try again shortly", status=503)
        try:
            detect_start = time.perf_counter()
            detection = detector.detect(frame)
            detect_ms = round((time.perf_counter() - detect_start) * 1000, 3)
        finally:
            detectors.release(detector)

        if detection is None:
            return jsonify({"hand_detected": False, "label": None, "confidence": 0.0, "top_k": [],
                            "backend": backend, "detect_ms": detect_ms, "cached": False,
                            "latency_ms": elapsed_ms()})

        result, cached = classify(to_features(detection.points), backend, top_k)
        return jsonify({"hand_detected": True, **result, **detection.to_dict(),
                        "detect_ms": detect_ms, "cached": cached, "latency_ms": elapsed_ms()})

    @app.get("/api/metrics")
    def get_metrics():
        snapshot = metrics.snapshot()
        snapshot["cache"] = cache.stats() if cache is not None else None
        snapshot["batching"] = batcher.stats() if batcher is not None else None
        snapshot["default_backend"] = predictor.default_backend
        snapshot["threads"] = app.config.get("SERVER_THREADS")
        return jsonify(snapshot)

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="ASL recognition REST API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--threads", default=8, type=int,
                        help="worker threads handling requests concurrently")
    parser.add_argument("--cache-size", default=4096, type=int)
    parser.add_argument("--no-cache", action="store_true", help="disable the response cache")
    parser.add_argument("--no-batching", action="store_true", help="disable request micro-batching")
    parser.add_argument("--backend", default="ensemble", choices=["rf", "mlp", "ensemble"])
    return parser.parse_args()


def main():
    from waitress import serve

    args = parse_args()
    app = create_app(
        predictor=Predictor(backend=args.backend),
        cache_size=0 if args.no_cache else args.cache_size,
        max_concurrent_frames=max(1, args.threads),
        warm_detectors=True,
        batching=not args.no_batching,
    )
    app.config["SERVER_THREADS"] = args.threads
    print(f"Serving on http://{args.host}:{args.port} ({args.threads} threads, "
          f"cache {'off' if args.no_cache else 'on'}, batching {'off' if args.no_batching else 'on'}, "
          f"backend {args.backend})")
    try:
        serve(app, host=args.host, port=args.port, threads=args.threads,
              connection_limit=max(100, args.threads * 16), ident="asl-api")
    finally:
        app.extensions["asl"]["detectors"].close()


if __name__ == "__main__":
    main()
