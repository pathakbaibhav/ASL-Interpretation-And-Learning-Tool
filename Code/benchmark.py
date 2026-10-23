"""
Load-tests the REST API and writes the results to models/benchmark.json.

Each configuration starts its own api.py server, then simulated clients
stream requests at it as fast as the server answers:

* landmarks - like the web app's browser engine: each client "holds" a
  random sign from data.pickle for 15 frames, adding MediaPipe-sized jitter
  to every frame, then moves to the next sign.
* frames    - like the server engine: JPEG photos of hands, each with fresh
  sensor-style noise so no two uploads are identical.

Configurations compared:
  baseline      1 process, 1 thread; no cache, no micro-batching
  threaded      1 process, 8 threads; cache + micro-batching (python api.py)
  multiprocess  4 gunicorn workers x 4 threads; cache + micro-batching

Usage: python benchmark.py [--duration 10] [--clients 1 8 32]
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "tests" / "fixtures"

CONFIGS = {
    "baseline": lambda port: [sys.executable, "api.py", "--port", str(port),
                              "--threads", "1", "--no-cache", "--no-batching"],
    "threaded": lambda port: [sys.executable, "api.py", "--port", str(port), "--threads", "8"],
    "multiprocess": lambda port: [sys.executable, "-m", "gunicorn", "-c", "gunicorn.conf.py",
                                  "--bind", f"127.0.0.1:{port}", "--workers", "4", "--threads", "4",
                                  "wsgi:app"],
}
HOLD_FRAMES = 15
JITTER = 0.0015  # Typical frame-to-frame landmark noise for a steady hand.


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(command):
    port = free_port()
    proc = subprocess.Popen(command(port), cwd=HERE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import requests

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/api/health", timeout=5).ok:
                return proc, url
        except requests.RequestException:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("Server didn't start")


def landmark_requests(seed):
    """Yields (path, kwargs) for a simulated webcam stream of landmarks."""
    with open(HERE / "data.pickle", "rb") as f:
        samples = pickle.load(f)["data"]
    rng = np.random.default_rng(seed)
    while True:
        base = np.asarray(samples[rng.integers(len(samples))], dtype=np.float32).reshape(21, 2)
        for _ in range(HOLD_FRAMES):
            points = base + rng.normal(0, JITTER, base.shape)
            yield "/api/predict", {"json": {"landmarks": points.round(5).tolist()}}


def frame_requests(seed):
    """Yields (path, kwargs) uploading noisy JPEGs of hands."""
    import cv2

    rng = np.random.default_rng(seed)
    images = [cv2.imread(str(p)) for p in sorted(FIXTURES.glob("*.jpg"))]
    # Pre-encode distinct noisy variants so the client isn't the bottleneck.
    variants = []
    for _ in range(40):
        img = images[rng.integers(len(images))].astype(np.int16)
        noisy = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        variants.append(cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    while True:
        for body in variants:
            yield "/api/predict/frame", {"data": body, "headers": {"Content-Type": "image/jpeg"}}


def client_worker(args):
    """Runs a few client threads in one process; returns their latencies and errors."""
    url, workload, threads, duration, seed = args
    import threading

    import requests

    latencies, errors, lock = [], [0], threading.Lock()
    stop_at = time.perf_counter() + duration

    def run(thread_seed):
        session = requests.Session()
        gen = (landmark_requests if workload == "landmarks" else frame_requests)(thread_seed)
        local, local_errors = [], 0
        for path, kwargs in gen:
            if time.perf_counter() >= stop_at:
                break
            start = time.perf_counter()
            try:
                ok = session.post(url + path, timeout=10, **kwargs).status_code == 200
            except requests.RequestException:
                ok = False
            if ok:
                local.append((time.perf_counter() - start) * 1000)
            else:
                local_errors += 1
        with lock:
            latencies.extend(local)
            errors[0] += local_errors

    pool = [threading.Thread(target=run, args=(seed * 100 + i,)) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    return latencies, errors[0]


def run_load(url, workload, clients, duration):
    """Spreads clients over processes so the load generator isn't GIL-bound."""
    processes = min(clients, os.cpu_count() or 4)
    per_process = [clients // processes + (i < clients % processes) for i in range(processes)]
    jobs = [(url, workload, n, duration, random.randrange(1 << 30)) for n in per_process]
    with mp.get_context("spawn").Pool(processes) as pool:
        results = pool.map(client_worker, jobs)
    latencies = np.concatenate([np.asarray(r[0]) for r in results]) if results else np.array([])
    errors = sum(r[1] for r in results)
    if not latencies.size:
        return {"requests": 0, "errors": errors}
    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    return {
        "requests": int(latencies.size),
        "errors": int(errors),
        "throughput_rps": round(latencies.size / duration, 1),
        "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", default=10, type=float, help="seconds per run")
    parser.add_argument("--clients", default=[1, 8, 32], type=int, nargs="+")
    parser.add_argument("--workloads", default=["landmarks", "frames"], nargs="+",
                        choices=["landmarks", "frames"])
    parser.add_argument("--configs", default=list(CONFIGS), nargs="+", choices=list(CONFIGS))
    parser.add_argument("--out", default=HERE / "models" / "benchmark.json", type=Path)
    args = parser.parse_args()

    import requests

    results = {"duration_s": args.duration, "cpu_count": os.cpu_count(), "runs": []}
    for config in args.configs:
        proc, url = start_server(CONFIGS[config])
        try:
            for workload in args.workloads:
                for clients in args.clients:
                    # Warm up so detector start-up and first-call costs aren't measured.
                    run_load(url, workload, min(clients, 4), 1.0)
                    stats = run_load(url, workload, clients, args.duration)
                    run = {"config": config, "workload": workload, "clients": clients, **stats}
                    results["runs"].append(run)
                    lat = stats.get("latency_ms", {})
                    print(f"{config:<10} {workload:<10} {clients:>3} clients  "
                          f"{stats.get('throughput_rps', 0):>8} req/s  p50 {lat.get('p50', '-'):>7} ms  "
                          f"p95 {lat.get('p95', '-'):>7} ms  errors {stats['errors']}", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=20)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
