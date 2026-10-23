// In-browser letter classifier for the static (GitHub Pages) build.
//
// A JavaScript port of features.normalized_features() and predictor.NumpyMLP:
// the TensorFlow-trained network's weights are exported to models/mlp.json
// by build_site.py, and tests/test_js_classifier.py checks this file gives
// the same answers as the Python code.

/** Shifts landmarks to start at (0, 0) and scales the longest side to 1. */
export function normalizeLandmarks(points) {
  if (!Array.isArray(points) || points.length !== 21) {
    throw new Error(`Expected 21 landmarks, got ${points?.length}`);
  }
  let minX = Infinity, minY = Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
  }
  let scale = 0;
  for (const [x, y] of points) scale = Math.max(scale, x - minX, y - minY);
  if (scale < 1e-6) throw new Error("Landmarks are degenerate");

  const features = new Float32Array(42);
  points.forEach(([x, y], i) => {
    features[2 * i] = (x - minX) / scale;
    features[2 * i + 1] = (y - minY) / scale;
  });
  return features;
}

export class MLPClassifier {
  /** @param model parsed mlp.json: {labels, mean, std, layers: [{W, b}]} */
  constructor(model) {
    this.labels = model.labels;
    this.mean = Float32Array.from(model.mean);
    this.std = Float32Array.from(model.std);
    this.layers = model.layers.map(({ W, b }) => ({
      rows: W.length,
      cols: b.length,
      W: Float32Array.from(W.flat()), // row-major, rows = inputs
      b: Float32Array.from(b),
    }));
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Couldn't load ${url}: ${response.status}`);
    return new MLPClassifier(await response.json());
  }

  /** Returns class probabilities for one set of normalized features. */
  predictProba(features) {
    let h = Float32Array.from(features, (v, i) => (v - this.mean[i]) / this.std[i]);
    this.layers.forEach((layer, n) => {
      const out = Float32Array.from(layer.b);
      for (let i = 0; i < layer.rows; i++) {
        const v = h[i];
        if (v === 0) continue;
        const offset = i * layer.cols;
        for (let j = 0; j < layer.cols; j++) out[j] += v * layer.W[offset + j];
      }
      if (n < this.layers.length - 1) {
        for (let j = 0; j < out.length; j++) out[j] = Math.max(out[j], 0);
      }
      h = out;
    });
    const max = Math.max(...h);
    const exp = h.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((v) => v / sum);
  }

  /** Same result shape as the REST API's /api/predict. */
  predict(points, topK = 3) {
    const proba = this.predictProba(normalizeLandmarks(points));
    const order = [...proba.keys()].sort((a, b) => proba[b] - proba[a]).slice(0, topK);
    const round = (v) => Math.round(v * 1e4) / 1e4;
    return {
      label: this.labels[order[0]],
      confidence: round(proba[order[0]]),
      top_k: order.map((i) => ({ label: this.labels[i], confidence: round(proba[i]) })),
      backend: "mlp",
    };
  }
}
