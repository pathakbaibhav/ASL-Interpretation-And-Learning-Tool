"""Thread-safe LRU cache used to reuse predictions for repeated inputs."""

import threading
from collections import OrderedDict

_MISSING = object()


class LRUCache:
    """
    A fixed-size least-recently-used cache that's safe to share between
    request threads. Tracks hits and misses for the /api/metrics endpoint.
    """

    def __init__(self, maxsize=4096):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        with self._lock:
            value = self._data.get(key, _MISSING)
            if value is _MISSING:
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def __len__(self):
        with self._lock:
            return len(self._data)

    def stats(self):
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / lookups, 4) if lookups else 0.0,
            }
