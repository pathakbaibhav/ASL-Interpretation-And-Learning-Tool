import numpy as np
import pytest

from features import (
    augment, legacy_features, mirror, normalized_features, normalized_from_legacy,
    quantize, to_points,
)

RNG = np.random.default_rng(0)
HAND = RNG.uniform(0.2, 0.6, size=(21, 2)).astype(np.float32)


def test_to_points_accepts_supported_shapes():
    flat = HAND.reshape(-1).tolist()
    dicts = [{"x": x, "y": y} for x, y in HAND]
    with_z = np.hstack([HAND, np.zeros((21, 1))]).tolist()
    for value in (HAND.tolist(), flat, dicts, with_z):
        np.testing.assert_allclose(to_points(value), HAND, rtol=1e-6)


@pytest.mark.parametrize("bad", [None, [], [[0, 0]] * 20, [[0, 0, 0, 0]] * 21, [["a", "b"]] * 21,
                                 [[float("nan"), 0]] * 21])
def test_to_points_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        to_points(bad)


def test_normalized_features_ignore_position_and_distance():
    base = normalized_features(HAND)
    moved = normalized_features(HAND * 0.5 + 0.3)  # further away and shifted
    np.testing.assert_allclose(base, moved, atol=1e-5)
    assert base.min() == 0 and np.isclose(base.max(), 1)


def test_legacy_round_trip_matches_v2():
    np.testing.assert_allclose(normalized_from_legacy(legacy_features(HAND)), normalized_features(HAND), atol=1e-6)


def test_degenerate_hand_is_rejected():
    with pytest.raises(ValueError):
        normalized_features(np.full((21, 2), 0.5))


def test_mirror_is_its_own_inverse():
    np.testing.assert_allclose(mirror(mirror(HAND)), HAND, atol=1e-6)


def test_augment_keeps_shape_and_stays_close():
    out = augment(HAND, np.random.default_rng(1), mirror_prob=0.0)
    assert out.shape == (21, 2)
    assert np.abs(out - HAND).max() < 0.2


def test_quantize_merges_tiny_differences():
    on_grid = quantize(normalized_features(HAND)) * 0.005
    assert quantize(on_grid).tobytes() == quantize(on_grid + 0.0004).tobytes()
    assert quantize(on_grid).tobytes() != quantize(on_grid + 0.004).tobytes()
