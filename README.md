# ASL Interpretation and Learning Tool

Real-time American Sign Language fingerspelling recognition (A–Z plus *space*) with a Flask REST API, a browser frontend, and a practice mode for learning the alphabet.

A webcam frame goes through **MediaPipe** hand tracking (21 landmarks), the landmarks are normalized, and an ensemble of a **scikit-learn Random Forest** and a **TensorFlow** neural network predicts the letter. Signs held steadily are assembled into words, and the browser can speak the result aloud.

---

## Highlights

| | |
|---|---|
| **Accuracy** | 100% on the held-out test split; stays at 100% when the hand is nearer or farther, rotated, or left-handed, where the original model fell to 88% / 89% / 54% |
| **Low light** | Retrying dark frames with automatic gamma correction raises hand detection from 33% to 64% at ≤25% brightness |
| **Latency** | 2.7 ms median per landmark request and 28 ms per uploaded frame (single client); about 25 ms end to end in the browser (17 ms on-device hand tracking + 7 ms API round trip) |
| **Throughput** | At 32 concurrent clients, 3.5× more landmark requests (443 → 1,569 req/s) and 7× more frame uploads (35 → 248 req/s) per second than a single-threaded server, with p95 latency down from 77 ms to 33 ms and from 946 ms to 194 ms |
| **Privacy** | In browser mode, video never leaves the device; only 42 numbers per frame are sent |

All numbers come from the scripts in this repo and are reproducible (see [Measuring it yourself](#measuring-it-yourself)).

---

## Architecture

```
 Browser (web/)                                    Flask API (api.py)
┌──────────────────────────────┐   POST /api/predict  ┌───────────────────────────────────────┐
│ webcam → MediaPipe (WASM/GPU)│ ── 21 landmarks ───▶ │ validate → normalize (features.py)     │
│ draw skeleton, hold-to-type, │                      │ → LRU response cache (cache.py)        │
│ speech, practice mode        │ ◀── label + top-3 ── │ → micro-batcher (batcher.py)           │
└──────────────────────────────┘                      │ → RF + TensorFlow MLP (predictor.py)   │
        or, as a fallback:                            │                                        │
 JPEG frame ── POST /api/predict/frame ─────────────▶ │ detector pool → MediaPipe + low-light  │
                                                      │ retry (hand_detector.py) → same path   │
                                                      └───────────────────────────────────────┘
                                                        waitress (threads) or gunicorn (processes)
```

**Model pipeline:** `collect_imgs.py` (webcam capture) → `create_dataset.py` (MediaPipe landmarks → `data.pickle`) → `train_classifier.py` (augmentation, Random Forest + TensorFlow MLP, evaluation) → `models/`.

### What each piece does

- **Feature normalization (`features.py`).** Landmarks are shifted to the hand's bounding box and scaled by its longest side, so predictions don't depend on where the hand is or how far it is from the camera.
- **Training augmentation.** Every training sample gets 5 randomly mirrored, rotated (±15°), scaled and jittered copies. Mirroring is why left-handed signing works.
- **Two models, one answer.** The Random Forest and the TensorFlow MLP average their probabilities (`ensemble`, the default). Either can be selected per request. The MLP is trained in TensorFlow, and its weights are exported and evaluated with NumPy at serving time: identical output (checked to within 1e-4 during training), 0.013 ms per prediction versus about 2 ms through Keras, and no TensorFlow dependency on the server.
- **Low-light fallback (`hand_detector.py`).** If no hand is found and the frame is dark, the frame is gamma-corrected toward mid-grey and detection runs again. Well-lit frames never pay for the second pass. Gamma alone beat CLAHE + gamma in testing, since CLAHE amplifies sensor noise.
- **Concurrency (`api.py`, `batcher.py`, `gunicorn.conf.py`).**
  - Requests are served by a pool of worker threads (waitress) or by several worker processes (gunicorn).
  - Concurrent predictions are coalesced into one vectorized model call by a micro-batcher. A lone request isn't delayed, because the batcher never waits for company.
  - MediaPipe detectors aren't thread-safe, so frames borrow one from a pre-warmed pool. When every detector stays busy for a second, the request gets HTTP 503 instead of queueing without limit.
- **Response cache (`cache.py`).** A thread-safe LRU cache keyed on features rounded to a 0.005 grid. Rounding was verified not to change any test-set prediction. It answers exact or near-exact repeats in about 0.2 ms. On live webcam streams, frame-to-frame landmark jitter means it rarely hits, so the throughput gains above come from batching and parallelism, not caching.
- **Observability.** `GET /api/metrics` reports per-endpoint request counts, errors, p50/p95/p99 latency, cache hit rate and average batch size. Every response carries a `Server-Timing` header, visible in browser devtools.

---

## Results

### Classification accuracy

From `models/metrics.json`, produced by `train_classifier.py`. The test split has 538 samples across 27 classes. "Baseline" is the original project's approach retrained on the same split: Random Forest on un-normalized landmarks, no augmentation. The harder variants perturb the same held-out samples.

| Test set | Baseline | Random Forest | TF MLP | Ensemble |
|---|---:|---:|---:|---:|
| Clean held-out split | 100.0% | 100.0% | 100.0% | 100.0% |
| Hand nearer/farther (×0.6–0.8 or ×1.2–1.4) | 87.9% | 100.0% | 100.0% | 100.0% |
| Rotated, rescaled, jittered | 88.7% | 99.8% | 100.0% | 100.0% |
| Left hand (mirrored) | 53.7% | 100.0% | 100.0% | 100.0% |

> **Caveat:** the dataset has about 100 consecutive webcam frames per letter from one signer and one session, so a random split puts near-duplicate frames on both sides. The clean split is easy for every model. Expect lower accuracy with new signers, backgrounds and cameras. Collecting data from more people is the most valuable next step.

### Low-light hand detection

From `models/lighting.json`, produced by `evaluate_lighting.py` on 9 hand images darkened step by step:

| Brightness | Mean pixel | Detected (plain) | Detected (with fallback) |
|---:|---:|---:|---:|
| 100% | 161.7 | 9/9 | 9/9 |
| 35% | 56.1 | 9/9 | 9/9 |
| 25% | 40.1 | 6/9 | 7/9 |
| 18% | 28.6 | 6/9 | 6/9 |
| 12% | 18.9 | 3/9 | 6/9 |
| 8% | 12.5 | 0/9 | 6/9 |
| 5% | 7.6 | 0/9 | 4/9 |

> Only 9 images (3 photos, plus mirrored and downscaled copies), so treat this as indicative. Run `python evaluate_lighting.py --images ./data` on your own captures for a stronger measurement.

### API latency and throughput

From `models/benchmark.json`, produced by `benchmark.py` on 10 CPU cores. Each run lasts 10 s with simulated webcam clients sending requests back to back. Landmark requests come from `data.pickle` with frame-to-frame jitter added; frame requests are noisy JPEGs of hands.

| Config | Workload | Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | Errors |
|---|---|---:|---:|---:|---:|---:|
| baseline | frames | 1 | 32.7 | 29.65 | 30.19 | 0 |
| threaded | frames | 1 | 34.7 | 27.93 | 29.88 | 0 |
| multiprocess | frames | 1 | 34.7 | 27.83 | 29.73 | 0 |
| baseline | frames | 8 | 34.8 | 227.57 | 228.74 | 0 |
| threaded | frames | 8 | 247.4 | 30.47 | 38.2 | 0 |
| multiprocess | frames | 8 | 196.8 | 38.09 | 58.72 | 0 |
| baseline | frames | 32 | 35.2 | 912.42 | 945.52 | 0 |
| threaded | frames | 32 | 216.3 | 137.31 | 161.47 | 0 |
| multiprocess | frames | 32 | 248.3 | 134.49 | 194.27 | 0 |
| baseline | landmarks | 1 | 362.5 | 2.74 | 3.08 | 0 |
| threaded | landmarks | 1 | 360.2 | 2.76 | 3.05 | 0 |
| multiprocess | landmarks | 1 | 323.4 | 3.04 | 3.43 | 0 |
| baseline | landmarks | 8 | 440.5 | 17.75 | 19.7 | 0 |
| threaded | landmarks | 8 | 840.1 | 9.12 | 14.41 | 0 |
| multiprocess | landmarks | 8 | 1,228.0 | 6.19 | 10.85 | 0 |
| baseline | landmarks | 32 | 442.7 | 71.99 | 77.25 | 0 |
| threaded | landmarks | 32 | 790.2 | 38.16 | 63.44 | 0 |
| multiprocess | landmarks | 32 | 1,569.3 | 18.72 | 32.89 | 0 |

- **baseline**: 1 process, 1 thread, no cache, no batching
- **threaded**: `python api.py` (8 threads, cache, micro-batching)
- **multiprocess**: `gunicorn -c gunicorn.conf.py wsgi:app` (4 workers × 4 threads, cache, micro-batching)

---

## Getting started

Requires Python 3.11 (MediaPipe and TensorFlow wheels lag behind the newest Python releases).

```bash
git clone https://github.com/pathakbaibhav/ASL-Interpretation-And-Learning-Tool.git
cd ASL-Interpretation-And-Learning-Tool
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Trained models are included in `Code/models/`. MediaPipe's hand landmarker model (7.8 MB) downloads automatically on first run.

### Web app and REST API

```bash
cd Code
python api.py
```

Open http://127.0.0.1:8000 and click **Start camera**.

- **Translate** mode: hold a sign until the bar fills to type it. Use Space/⌫ or the keyboard to edit, and **Speak** to read the text aloud.
- **Practice** mode: shows a reference handshape, checks your sign, and tracks streaks and mastered letters.
- **Engine**: the default runs MediaPipe in the browser. Choose *Server frames* to upload frames instead.

For a multi-process deployment (Linux/macOS):

```bash
cd Code
gunicorn -c gunicorn.conf.py wsgi:app
```

### OpenCV desktop demo

To check the model without a browser, this opens a window showing the webcam, the hand skeleton and the predicted letter (press Q to quit):

```bash
cd Code
python inference_classifier.py
```

---

## REST API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/health` | | status, backends, labels |
| `POST` | `/api/predict` | `{"landmarks": [[x, y] × 21], "backend"?: "ensemble"\|"rf"\|"mlp", "top_k"?: 1–5}` | `label`, `confidence`, `top_k`, `cached`, `latency_ms` |
| `POST` | `/api/predict/batch` | `{"items": [landmarks, …]}` (max 256) | `results` in order |
| `POST` | `/api/predict/frame` | a JPEG/PNG body, a multipart `image` field, or `{"image": "<base64 or data URL>"}` | the prediction plus `hand_detected`, `landmarks`, `handedness`, `lighting_corrected`, `detect_ms` |
| `GET` | `/api/metrics` | | request counts, latency percentiles, cache and batching stats |

Landmarks may be `[x, y]` pairs, `{"x", "y"}` objects, `[x, y, z]` triples or a flat list of 42 numbers, in MediaPipe's normalized image coordinates. Errors return a 4xx status with `{"error": "…"}`. Overload returns 503.

```bash
curl -s -X POST localhost:8000/api/predict/frame \
  -H 'Content-Type: image/jpeg' --data-binary @Code/tests/fixtures/victory.jpg
```

---

## Measuring it yourself

```bash
pip install -r requirements-dev.txt
cd Code
python -m pytest tests
python train_classifier.py
python evaluate_lighting.py
python benchmark.py
```

The commands do the following, in order:

1. `pytest`: unit and API tests.
2. `train_classifier.py`: retrains both models and writes `models/metrics.json`.
3. `evaluate_lighting.py`: low-light detection, written to `models/lighting.json`. Pass `--images ./data` to use your own captures.
4. `benchmark.py`: the load test, written to `models/benchmark.json`.

### Training on your own data

```bash
cd Code
python collect_imgs.py
python create_dataset.py
python train_classifier.py
```

1. `collect_imgs.py` captures 100 webcam frames per letter into `./data/<letter>/`.
2. `create_dataset.py` extracts landmarks into `data.pickle`.
3. `train_classifier.py` trains the models and writes them to `models/`.

---

## Project structure

```
Code/
├── api.py                 Flask REST API + web app (waitress server)
├── wsgi.py, gunicorn.conf.py   multi-process deployment
├── batcher.py             dynamic micro-batching of predictions
├── cache.py               thread-safe LRU response cache
├── metrics.py             latency percentiles for /api/metrics
├── predictor.py           loads models; RF / NumPy-served TF MLP / ensemble
├── hand_detector.py       MediaPipe HandLandmarker + low-light fallback
├── features.py            landmark normalization and augmentation
├── train_classifier.py    training + evaluation → models/metrics.json
├── evaluate_lighting.py   low-light detection study → models/lighting.json
├── benchmark.py           load test → models/benchmark.json
├── collect_imgs.py        capture training images from the webcam
├── create_dataset.py      images → landmarks (data.pickle)
├── inference_classifier.py  OpenCV desktop demo
├── data.pickle            landmark dataset (2,688 samples, 27 classes)
├── models/                trained models and measured results
├── signImages/            reference handshapes A–Z
├── web/                   HTML/CSS/JS frontend
└── tests/                 pytest suite
```

## Tools and technologies

Python · Flask · waitress / gunicorn · TensorFlow (Keras) · scikit-learn · MediaPipe (Python and Web/WASM) · OpenCV · NumPy · HTML5 / CSS3 / JavaScript (getUserMedia, Canvas, Web Speech API) · pytest
