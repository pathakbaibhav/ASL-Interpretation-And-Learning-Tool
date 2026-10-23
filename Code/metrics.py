"""Per-endpoint request counters and latency percentiles for /api/metrics."""

import threading
import time
from collections import defaultdict, deque

import numpy as np


class RequestMetrics:
    """Records the latency of recent requests per endpoint; thread-safe."""

    def __init__(self, window=2048):
        self._lock = threading.Lock()
        self._latencies = defaultdict(lambda: deque(maxlen=window))
        self._counts = defaultdict(int)
        self._errors = defaultdict(int)
        self._started = time.time()

    def record(self, endpoint, latency_ms, error=False):
        with self._lock:
            self._counts[endpoint] += 1
            if error:
                self._errors[endpoint] += 1
            else:
                self._latencies[endpoint].append(latency_ms)

    def snapshot(self):
        with self._lock:
            endpoints = {}
            for name, count in self._counts.items():
                samples = np.asarray(self._latencies[name])
                summary = {"requests": count, "errors": self._errors[name]}
                if samples.size:
                    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
                    summary["latency_ms"] = {
                        "p50": round(float(p50), 3),
                        "p95": round(float(p95), 3),
                        "p99": round(float(p99), 3),
                        "max": round(float(samples.max()), 3),
                        "window": int(samples.size),
                    }
                endpoints[name] = summary
            return {"uptime_s": round(time.time() - self._started, 1), "endpoints": endpoints}
