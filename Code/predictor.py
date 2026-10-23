"""
Loads the trained classifiers and turns v2 hand features into letter
predictions.

Three backends are available:

* ``rf``       - scikit-learn Random Forest.
* ``mlp``      - a neural network trained with TensorFlow/Keras. Its weights
                 are exported to ``mlp.npz`` and evaluated with NumPy at
                 serving time, which gives the same output as Keras in a
                 fraction of the time and keeps TensorFlow out of the server.
* ``ensemble`` - the average of both probability distributions (default).

All backends are safe to call from many threads at once.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from features import NUM_FEATURES

MODELS_DIR = Path(__file__).resolve().parent / "models"
BACKENDS = ("rf", "mlp", "ensemble")


class NumpyMLP:
    """Forward pass of the exported Keras network: standardize, ReLU layers, softmax."""

    def __init__(self, path):
        weights = np.load(path)
        self.mean = weights["mean"]
        self.std = weights["std"]
        self.layers = []
        i = 0
        while f"W{i}" in weights:
            self.layers.append((weights[f"W{i}"], weights[f"b{i}"]))
            i += 1

    def predict_proba(self, X):
        h = (np.asarray(X, dtype=np.float32) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        h = h - h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)


class Predictor:
    def __init__(self, models_dir=MODELS_DIR, backend="ensemble"):
        models_dir = Path(models_dir)
        meta = json.loads((models_dir / "labels.json").read_text())
        if meta.get("feature_version") != 2:
            raise ValueError("Models were trained on an unsupported feature version; retrain them")
        self.labels = np.array(meta["labels"])

        self.rf = joblib.load(models_dir / "rf.joblib")
        # Parallelism across trees costs more than it saves for one sample.
        self.rf.n_jobs = 1
        if list(self.rf.classes_) != list(self.labels):
            raise ValueError("rf.joblib classes don't match labels.json")

        mlp_path = models_dir / "mlp.npz"
        self.mlp = NumpyMLP(mlp_path) if mlp_path.exists() else None

        self.default_backend = self._check_backend(backend)

    @property
    def backends(self):
        return BACKENDS if self.mlp is not None else ("rf",)

    def _check_backend(self, backend):
        if backend not in self.backends:
            raise ValueError(f"Unknown or unavailable backend {backend!r}; choose from {self.backends}")
        return backend

    def predict_proba(self, X, backend=None):
        """Returns class probabilities, shape (n_samples, n_labels)."""
        backend = self._check_backend(backend or self.default_backend)
        X = np.asarray(X, dtype=np.float32).reshape(-1, NUM_FEATURES)
        if backend == "rf":
            return self.rf.predict_proba(X)
        if backend == "mlp":
            return self.mlp.predict_proba(X)
        return (self.rf.predict_proba(X) + self.mlp.predict_proba(X)) / 2

    def predict(self, features, backend=None, top_k=3):
        """Predicts one sample. Returns a dict with the label, confidence and runners-up."""
        return self.predict_batch([features], backend, top_k)[0]

    def predict_batch(self, features_list, backend=None, top_k=3):
        """Predicts many samples in one vectorized call."""
        backend = backend or self.default_backend
        proba = self.predict_proba(np.asarray(features_list), backend)
        return [self.format_result(row, backend, top_k) for row in proba]

    def format_result(self, proba_row, backend, top_k=3):
        """Turns one row of class probabilities into the API's result dict."""
        idx = np.argsort(-proba_row)[:top_k]
        return {
            "label": str(self.labels[idx[0]]),
            "confidence": round(float(proba_row[idx[0]]), 4),
            "top_k": [
                {"label": str(self.labels[j]), "confidence": round(float(proba_row[j]), 4)}
                for j in idx
            ],
            "backend": backend,
        }
