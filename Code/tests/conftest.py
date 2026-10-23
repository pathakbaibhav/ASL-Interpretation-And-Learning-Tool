import pickle
import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(CODE_DIR))


@pytest.fixture(scope="session")
def dataset():
    path = CODE_DIR / "data.pickle"
    if not path.exists():
        pytest.skip("data.pickle not available")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def predictor():
    from predictor import Predictor

    return Predictor()


@pytest.fixture(scope="session")
def app(predictor):
    from api import create_app

    app = create_app(predictor=predictor, cache_size=256, max_concurrent_frames=2)
    yield app
    app.extensions["asl"]["detectors"].close()


@pytest.fixture()
def client(app):
    app.extensions["asl"]["cache"].clear()
    return app.test_client()


@pytest.fixture(scope="session")
def victory_jpg():
    """A right hand making a "V" (MediaPipe sample image, Apache 2.0)."""
    return (FIXTURES / "victory.jpg").read_bytes()
