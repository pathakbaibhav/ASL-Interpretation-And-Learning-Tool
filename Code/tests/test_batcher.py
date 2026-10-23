import threading

import numpy as np
import pytest

from batcher import MicroBatcher
from features import normalized_from_legacy


def test_concurrent_requests_get_their_own_answers(predictor, dataset):
    batcher = MicroBatcher(predictor)
    idx = list(range(0, len(dataset["data"]), 25))[:64]
    features = [normalized_from_legacy(dataset["data"][i]) for i in idx]
    expected = [predictor.predict(f, "ensemble")["label"] for f in features]
    results = [None] * len(features)

    def worker(i):
        results[i] = batcher.predict(features[i], "ensemble", 3)["label"]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(features))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == expected
    assert batcher.stats()["items"] == len(features)


def test_mixed_backends_in_one_batch(predictor, dataset):
    batcher = MicroBatcher(predictor)
    f = normalized_from_legacy(dataset["data"][0])
    assert batcher.predict(f, "rf", 1)["backend"] == "rf"
    assert batcher.predict(f, "mlp", 2)["backend"] == "mlp"


def test_errors_reach_the_caller(predictor):
    batcher = MicroBatcher(predictor)
    with pytest.raises(ValueError):
        batcher.predict(np.zeros(42), "svm", 3)
