import numpy as np
import pytest

from features import normalized_from_legacy


@pytest.fixture(scope="module")
def samples(dataset):
    idx = np.random.default_rng(0).choice(len(dataset["data"]), 200, replace=False)
    X = np.stack([normalized_from_legacy(dataset["data"][i]) for i in idx])
    y = np.asarray(dataset["labels"])[idx]
    return X, y


@pytest.mark.parametrize("backend", ["rf", "mlp", "ensemble"])
def test_backends_classify_dataset_samples(predictor, samples, backend):
    X, y = samples
    predicted = predictor.labels[predictor.predict_proba(X, backend).argmax(1)]
    assert (predicted == y).mean() > 0.95


def test_probabilities_are_distributions(predictor, samples):
    proba = predictor.predict_proba(samples[0], "ensemble")
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert proba.shape == (len(samples[0]), len(predictor.labels))


def test_batch_matches_single(predictor, samples):
    X = samples[0][:10]
    batch = predictor.predict_batch(X)
    single = [predictor.predict(x) for x in X]
    assert [b["label"] for b in batch] == [s["label"] for s in single]


def test_top_k_is_sorted(predictor, samples):
    result = predictor.predict(samples[0][0], top_k=5)
    confidences = [t["confidence"] for t in result["top_k"]]
    assert len(confidences) == 5 and confidences == sorted(confidences, reverse=True)
    assert result["label"] == result["top_k"][0]["label"]


def test_unknown_backend_raises(predictor, samples):
    with pytest.raises(ValueError):
        predictor.predict(samples[0][0], backend="svm")
