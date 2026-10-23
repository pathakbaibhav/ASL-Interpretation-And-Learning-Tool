from pathlib import Path

import cv2
import numpy as np
import pytest

from hand_detector import HandDetector, enhance_lighting, frame_brightness


@pytest.fixture(scope="module")
def frame(victory_jpg):
    return cv2.imdecode(np.frombuffer(victory_jpg, np.uint8), cv2.IMREAD_COLOR)


def test_detects_a_hand(frame):
    with HandDetector() as detector:
        detection = detector.detect(frame)
    assert detection is not None
    assert detection.points.shape == (21, 2)
    assert not detection.lighting_corrected


def test_no_hand_in_blank_frame():
    with HandDetector() as detector:
        assert detector.detect(np.zeros((240, 320, 3), np.uint8)) is None


def test_low_light_fallback_recovers_dark_frame():
    from conftest import FIXTURES

    frame = cv2.imread(str(FIXTURES / "pointing_up.jpg"))
    dark = (frame.astype(np.float32) * 0.08).astype(np.uint8)
    with HandDetector(lighting="off") as plain, HandDetector(lighting="auto") as adaptive:
        assert plain.detect(dark) is None
        detection = adaptive.detect(dark)
    assert detection is not None and detection.lighting_corrected


def test_enhance_lighting_brightens_dark_frames(frame):
    dark = (frame.astype(np.float32) * 0.1).astype(np.uint8)
    assert frame_brightness(enhance_lighting(dark)) > 3 * frame_brightness(dark)


def test_video_mode_requires_timestamps(frame):
    with HandDetector(video=True) as detector:
        with pytest.raises(ValueError):
            detector.detect(frame)
        assert detector.detect(frame, timestamp_ms=0) is not None
        # Repeated or decreasing timestamps are bumped instead of crashing.
        assert detector.detect(frame, timestamp_ms=0) is not None


def test_model_download_is_atomic(tmp_path, monkeypatch):
    import model_files

    def fake_download(url, filename):
        Path(filename).write_bytes(b"model")

    monkeypatch.setattr(model_files.urllib.request, "urlretrieve", fake_download)
    target = tmp_path / "models" / "hand_landmarker.task"
    assert model_files.ensure_landmarker_model(target) == target
    assert target.read_bytes() == b"model"
    assert [p.name for p in target.parent.iterdir()] == [target.name]  # no leftover temp files


def test_failed_download_leaves_nothing_behind(tmp_path, monkeypatch):
    import model_files

    def broken_download(url, filename):
        Path(filename).write_bytes(b"partial")
        raise OSError("network down")

    monkeypatch.setattr(model_files.urllib.request, "urlretrieve", broken_download)
    target = tmp_path / "hand_landmarker.task"
    with pytest.raises(OSError):
        model_files.ensure_landmarker_model(target)
    assert list(tmp_path.iterdir()) == []
