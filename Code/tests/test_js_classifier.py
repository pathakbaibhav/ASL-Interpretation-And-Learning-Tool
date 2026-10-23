"""Checks the browser classifier (classifier.js) matches the Python predictor."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from build_site import export_mlp_json
from conftest import CODE_DIR
from features import normalized_features

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")

def test_browser_classifier_matches_python(tmp_path, predictor, dataset):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dataset["data"]), 150, replace=False)
    # Landmarks in image coordinates, as MediaPipe in the browser reports them.
    samples = [(np.asarray(dataset["data"][i]).reshape(21, 2) * 0.8 + 0.1).tolist() for i in idx]

    shutil.copy(CODE_DIR / "web" / "static" / "js" / "classifier.js", tmp_path / "classifier.mjs")
    (tmp_path / "mlp.json").write_text(json.dumps(export_mlp_json(CODE_DIR / "models" / "mlp.npz",
                                                                  list(predictor.labels))))
    (tmp_path / "samples.json").write_text(json.dumps(samples))
    (tmp_path / "run.mjs").write_text("""
import { readFileSync } from "node:fs";
import { MLPClassifier, normalizeLandmarks } from "./classifier.mjs";
const clf = new MLPClassifier(JSON.parse(readFileSync("mlp.json", "utf8")));
const samples = JSON.parse(readFileSync("samples.json", "utf8"));
console.log(JSON.stringify(samples.map((points) => ({
  label: clf.predict(points).label,
  features: Array.from(normalizeLandmarks(points)),
  proba: Array.from(clf.predictProba(normalizeLandmarks(points))),
}))));
""")
    output = subprocess.run([NODE, "run.mjs"], cwd=tmp_path, capture_output=True, text=True, check=True)
    js = json.loads(output.stdout)

    for points, result in zip(samples, js):
        features = normalized_features(points)
        np.testing.assert_allclose(result["features"], features, atol=1e-6)
        expected = predictor.predict_proba(features, "mlp")[0]
        np.testing.assert_allclose(result["proba"], expected, atol=1e-4)
        assert result["label"] == predictor.labels[expected.argmax()]


def test_browser_classifier_rejects_bad_input(tmp_path):
    shutil.copy(CODE_DIR / "web" / "static" / "js" / "classifier.js", tmp_path / "classifier.mjs")
    (tmp_path / "run.mjs").write_text("""
import { normalizeLandmarks } from "./classifier.mjs";
for (const bad of [[], Array(21).fill([0.5, 0.5])]) {
  try { normalizeLandmarks(bad); console.log("accepted"); } catch { console.log("rejected"); }
}
""")
    output = subprocess.run([NODE, "run.mjs"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert output.stdout.split() == ["rejected", "rejected"]
