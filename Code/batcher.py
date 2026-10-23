"""
Dynamic micro-batching for prediction requests.

Request threads hand their features to a single inference thread, which
runs everything queued at that moment as one vectorized model call. A Random
Forest takes about as long to classify 32 hands as one, so under concurrent
load this multiplies throughput, while a lone request still goes straight
through without waiting for company.
"""

import queue
import threading
from collections import defaultdict
from concurrent.futures import Future

import numpy as np


class MicroBatcher:
    def __init__(self, predictor, max_batch=64):
        self.predictor = predictor
        self.max_batch = max_batch
        self._queue = queue.Queue()
        self.batches = 0
        self.items = 0
        self._thread = threading.Thread(target=self._run, name="micro-batcher", daemon=True)
        self._thread.start()

    def predict(self, features, backend, top_k, timeout=10):
        """Blocks until this sample's prediction is ready and returns it."""
        future = Future()
        self._queue.put((features, backend, top_k, future))
        return future.result(timeout=timeout)

    def _run(self):
        while True:
            batch = [self._queue.get()]
            # Take whatever else is already waiting; never wait for more.
            while len(batch) < self.max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._process(batch)

    def _process(self, batch):
        self.batches += 1
        self.items += len(batch)
        groups = defaultdict(list)
        for item in batch:
            groups[item[1]].append(item)
        for backend, items in groups.items():
            try:
                proba = self.predictor.predict_proba(np.stack([i[0] for i in items]), backend)
                for (_, _, top_k, future), row in zip(items, proba):
                    future.set_result(self.predictor.format_result(row, backend, top_k))
            except Exception as e:  # Hand the error to every waiting request.
                for item in items:
                    if not item[3].done():
                        item[3].set_exception(e)

    def stats(self):
        return {
            "batches": self.batches,
            "items": self.items,
            "mean_batch_size": round(self.items / self.batches, 2) if self.batches else 0.0,
        }
