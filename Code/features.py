"""
Feature extraction shared by training, the Flask API and the OpenCV demo,
so every entry point turns hand landmarks into model input
in exactly the same way.

Two feature layouts exist:

* v1 ("legacy"): the 21 (x, y) landmarks shifted so the hand's bounding box
  starts at (0, 0). This is what the original project used and what
  ``data.pickle`` stores.
* v2: v1 additionally divided by the bounding box's longest side, so the
  features no longer depend on how far the hand is from the camera.
"""

import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 2


def to_points(landmarks):
    """
    Converts landmarks in any supported shape into a (21, 2) float array.

    Accepts MediaPipe landmark objects (anything with ``.x`` and ``.y``),
    dicts with ``x``/``y`` keys, [x, y] pairs, or a flat list of 42 numbers.
    Raises ValueError for anything else.
    """
    if landmarks is None:
        raise ValueError("No landmarks given")

    first = landmarks[0] if len(landmarks) else None
    if hasattr(first, "x") and hasattr(first, "y"):
        points = [[lm.x, lm.y] for lm in landmarks]
    elif isinstance(first, dict):
        points = [[lm["x"], lm["y"]] for lm in landmarks]
    else:
        points = landmarks

    try:
        points = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError, KeyError) as e:
        raise ValueError(f"Landmarks are not numeric: {e}") from e

    if points.size == NUM_FEATURES:
        points = points.reshape(NUM_LANDMARKS, -1)
    if points.shape == (NUM_LANDMARKS, 3):
        points = points[:, :2]  # Drop the z coordinate if one was sent.
    if points.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            f"Expected {NUM_LANDMARKS} landmarks with x and y, got shape {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Landmarks contain NaN or infinite values")
    return points


def legacy_features(points):
    """Returns v1 features: landmarks shifted to start at (0, 0), flattened."""
    points = to_points(points)
    return (points - points.min(axis=0)).reshape(-1)


def normalized_features(points):
    """
    Returns v2 features: landmarks shifted to start at (0, 0) and scaled so
    the longest side of the hand's bounding box is 1.
    """
    shifted = to_points(points)
    shifted = shifted - shifted.min(axis=0)
    scale = float(shifted.max())
    if scale < 1e-6:
        raise ValueError("Landmarks are degenerate (all in one place)")
    return (shifted / scale).reshape(-1)


def normalized_from_legacy(legacy):
    """Converts stored v1 features (as in data.pickle) into v2 features."""
    return normalized_features(np.asarray(legacy, dtype=np.float32))


def mirror(points):
    """Mirrors a hand horizontally, turning a right hand into a left one."""
    points = to_points(points).copy()
    points[:, 0] = 1.0 - points[:, 0]
    return points


def augment(points, rng, mirror_prob=0.5, max_rotation_deg=15.0,
            scale_range=0.1, jitter=0.01):
    """
    Returns a randomly perturbed copy of a hand: optional mirroring, a small
    rotation about the hand's centre, non-uniform scaling and per-landmark
    jitter. Used to make the classifier robust to left-handed signers,
    camera angle and landmark noise.
    """
    points = to_points(points).copy()
    if rng.random() < mirror_prob:
        points = mirror(points)

    centre = points.mean(axis=0)
    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    scale = rng.uniform(1 - scale_range, 1 + scale_range, size=2).astype(np.float32)

    points = (points - centre) @ rotation.T * scale + centre
    points += rng.normal(0, jitter, size=points.shape).astype(np.float32)
    return points


def quantize(features, step=0.005):
    """Rounds features onto a grid; used to build response-cache keys."""
    return np.round(np.asarray(features) / step).astype(np.int16)
