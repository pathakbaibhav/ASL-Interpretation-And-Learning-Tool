import base64
import threading
import time

import pytest

from api import create_app


@pytest.fixture(scope="module")
def landmarks(dataset):
    return dataset["data"][0], dataset["labels"][0]


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert set(body["backends"]) == {"rf", "mlp", "ensemble"}
    assert len(body["labels"]) == 27


def test_predict_returns_label_and_caches(client, landmarks):
    points, label = landmarks
    first = client.post("/api/predict", json={"landmarks": points})
    assert first.status_code == 200
    body = first.get_json()
    assert body["label"] == label and body["cached"] is False
    assert 0 <= body["confidence"] <= 1 and len(body["top_k"]) == 3
    assert "Server-Timing" in first.headers

    second = client.post("/api/predict", json={"landmarks": points}).get_json()
    assert second["cached"] is True and second["label"] == label


@pytest.mark.parametrize("backend", ["rf", "mlp", "ensemble"])
def test_predict_each_backend(client, landmarks, backend):
    points, label = landmarks
    body = client.post("/api/predict", json={"landmarks": points, "backend": backend, "top_k": 1}).get_json()
    assert body["label"] == label and body["backend"] == backend and len(body["top_k"]) == 1


@pytest.mark.parametrize("payload, message", [
    ({}, "Missing 'landmarks'"),
    ({"landmarks": [[0, 0]] * 5}, "Invalid landmarks"),
    ({"landmarks": [[0.5, 0.5]] * 21}, "degenerate"),
    ({"landmarks": [[0, 0]] * 21, "backend": "svm"}, "Unknown backend"),
    ({"landmarks": [[0, 0]] * 21, "top_k": "x"}, "top_k"),
])
def test_predict_rejects_bad_input(client, payload, message):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_predict_requires_json(client):
    response = client.post("/api/predict", data="not json")
    assert response.status_code == 400


def test_batch(client, dataset):
    items = dataset["data"][:20]
    body = client.post("/api/predict/batch", json={"items": items}).get_json()
    assert body["count"] == 20
    assert [r["label"] for r in body["results"]] == dataset["labels"][:20]


def test_batch_limits(client, dataset):
    assert client.post("/api/predict/batch", json={"items": []}).status_code == 400
    too_many = client.post("/api/predict/batch", json={"items": [dataset["data"][0]] * 257})
    assert too_many.status_code == 400
    bad_item = client.post("/api/predict/batch", json={"items": [dataset["data"][0], [1, 2]]})
    assert "item 1" in bad_item.get_json()["error"]


def test_frame_raw_jpeg(client, victory_jpg):
    body = client.post("/api/predict/frame", data=victory_jpg, content_type="image/jpeg").get_json()
    assert body["hand_detected"] is True
    assert body["label"] == "V" and body["confidence"] > 0.8
    assert len(body["landmarks"]) == 21

    again = client.post("/api/predict/frame", data=victory_jpg, content_type="image/jpeg").get_json()
    assert again["cached"] is True


def test_frame_base64_json_and_multipart(client, victory_jpg):
    data_url = "data:image/jpeg;base64," + base64.b64encode(victory_jpg).decode()
    body = client.post("/api/predict/frame", json={"image": data_url, "backend": "rf"}).get_json()
    assert body["label"] == "V" and body["backend"] == "rf"

    from io import BytesIO
    body = client.post("/api/predict/frame", data={"image": (BytesIO(victory_jpg), "v.jpg")},
                       content_type="multipart/form-data").get_json()
    assert body["label"] == "V"


def test_frame_without_hand(client):
    import cv2
    import numpy as np

    ok, blank = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
    body = client.post("/api/predict/frame", data=blank.tobytes(), content_type="image/jpeg").get_json()
    assert body["hand_detected"] is False and body["label"] is None


@pytest.mark.parametrize("kwargs", [
    {"data": b"not an image", "content_type": "image/jpeg"},
    {"json": {"image": "%%%not base64%%%"}},
    {"json": {"nothing": 1}},
])
def test_frame_rejects_bad_images(client, kwargs):
    assert client.post("/api/predict/frame", **kwargs).status_code == 400


def test_frame_too_large(client):
    response = client.post("/api/predict/frame", data=b"0" * (5 * 1024 * 1024), content_type="image/jpeg")
    assert response.status_code == 413


def test_busy_server_returns_503(predictor, victory_jpg):
    release = threading.Event()

    class SlowDetector:
        def detect(self, frame):
            release.wait(5)
            return None

    app = create_app(predictor=predictor, cache_size=0, max_concurrent_frames=1,
                     detector_factory=SlowDetector)
    blocker = threading.Thread(target=lambda: app.test_client().post(
        "/api/predict/frame", data=victory_jpg, content_type="image/jpeg"))
    blocker.start()
    time.sleep(0.2)
    try:
        response = app.test_client().post("/api/predict/frame", data=victory_jpg, content_type="image/jpeg")
        assert response.status_code == 503
    finally:
        release.set()
        blocker.join()


def test_metrics(client, landmarks):
    client.post("/api/predict", json={"landmarks": landmarks[0]})
    body = client.get("/api/metrics").get_json()
    assert body["endpoints"]["predict"]["requests"] >= 1
    assert "p95" in body["endpoints"]["predict"]["latency_ms"]
    assert body["cache"]["maxsize"] == 256


def test_cache_can_be_disabled(predictor, landmarks):
    client = create_app(predictor=predictor, cache_size=0).test_client()
    for _ in range(2):
        assert client.post("/api/predict", json={"landmarks": landmarks[0]}).get_json()["cached"] is False
    assert client.get("/api/metrics").get_json()["cache"] is None


def test_pages_and_assets(client):
    page = client.get("/")
    assert page.status_code == 200 and b"ASL Interpreter" in page.data
    assert client.get("/signs/A.png").mimetype == "image/png"
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/nope").get_json() == {"error": "Not found"}
