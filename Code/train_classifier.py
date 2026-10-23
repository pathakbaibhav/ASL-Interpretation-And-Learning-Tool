"""
Trains the letter classifiers and measures them.

Produces, in ./models:
  rf.joblib     Random Forest (scikit-learn)
  mlp.keras     neural network (TensorFlow/Keras)
  mlp.npz       the same network's weights, served with NumPy by predictor.py
  labels.json   class names and feature version
  metrics.json  accuracy, robustness and latency numbers quoted in the README

The original project (Random Forest on un-normalized v1 features, no
augmentation) is retrained on the same split as a baseline, so every
improvement is measured rather than assumed.

Usage: python train_classifier.py [--augment 5] [--rf-trees 50]
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from features import NUM_FEATURES, augment, normalized_features, quantize, to_points

HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=HERE / "data.pickle", type=Path)
    parser.add_argument("--models-dir", default=HERE / "models", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--augment", default=5, type=int,
                        help="augmented copies of every training sample")
    parser.add_argument("--rf-trees", default=50, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    return parser.parse_args()


def perturb_set(points_list, kind, rng):
    """Builds a harder copy of the test set to measure robustness."""
    out = []
    for p in points_list:
        p = p.copy()
        if kind == "distance":
            # Hand nearer to or further from the camera than in training.
            p = p * rng.choice([rng.uniform(0.6, 0.8), rng.uniform(1.2, 1.4)])
        elif kind == "pose":
            p = augment(p, rng, mirror_prob=0.0)
        elif kind == "left_hand":
            p[:, 0] = 1.0 - p[:, 0]
        out.append(p)
    return out


def build_mlp(n_classes, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_mlp(model, mean, std, path):
    """Saves Dense layer weights so predictor.py can run the network in NumPy."""
    import tensorflow as tf

    arrays = {"mean": mean, "std": std}
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    for i, layer in enumerate(dense):
        W, b = layer.get_weights()
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    np.savez(path, **arrays)


def single_sample_latency_ms(predict_fn, X, n=300):
    """Median time to classify one sample, in milliseconds."""
    times = []
    for i in range(n):
        x = X[i % len(X)][None, :]
        t = time.perf_counter()
        predict_fn(x)
        times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times)), 3)


def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    rng = np.random.default_rng(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # data.pickle stores v1 features: landmarks shifted to start at (0, 0).
    with open(args.data, "rb") as f:
        data_dict = pickle.load(f)
    points = [to_points(x) for x in data_dict["data"]]
    labels = np.asarray(data_dict["labels"])
    class_names = sorted(set(labels))
    print(f"Loaded {len(points)} samples across {len(class_names)} classes")

    idx_train, idx_test = train_test_split(
        np.arange(len(points)), test_size=0.2, shuffle=True,
        stratify=labels, random_state=args.seed,
    )
    train_pts = [points[i] for i in idx_train]
    test_pts = [points[i] for i in idx_test]
    y_train, y_test = labels[idx_train], labels[idx_test]

    test_sets = {
        "clean": test_pts,
        "distance": perturb_set(test_pts, "distance", rng),
        "pose": perturb_set(test_pts, "pose", rng),
        "left_hand": perturb_set(test_pts, "left_hand", rng),
    }

    def v1(pts):
        return np.stack([(p - p.min(axis=0)).reshape(-1) for p in pts])

    def v2(pts):
        return np.stack([normalized_features(p) for p in pts])

    # --- Baseline: the original project's model ------------------------------
    print("Training baseline (original v1 features, no augmentation) ...")
    baseline = RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    baseline.fit(v1(train_pts), y_train)

    # --- Improved models: v2 features + augmentation ----------------------------
    aug_pts, aug_y = list(train_pts), list(y_train)
    for p, y in zip(train_pts, y_train):
        for _ in range(args.augment):
            aug_pts.append(augment(p, rng))
            aug_y.append(y)
    X_train, y_train_aug = v2(aug_pts), np.asarray(aug_y)
    print(f"Training set: {len(train_pts)} samples, {len(X_train)} after augmentation")

    print(f"Training Random Forest ({args.rf_trees} trees) ...")
    rf = RandomForestClassifier(n_estimators=args.rf_trees, random_state=args.seed, n_jobs=-1)
    rf.fit(X_train, y_train_aug)
    rf.n_jobs = 1

    print("Training TensorFlow MLP ...")
    import tensorflow as tf

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    mean = X_train.mean(axis=0).astype(np.float32)
    std = (X_train.std(axis=0) + 1e-6).astype(np.float32)
    mlp = build_mlp(len(class_names), args.seed)
    mlp.fit(
        (X_train - mean) / std, np.array([label_to_idx[y] for y in y_train_aug]),
        validation_split=0.1, epochs=args.epochs, batch_size=64, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )
    mlp.save(args.models_dir / "mlp.keras")
    export_mlp(mlp, mean, std, args.models_dir / "mlp.npz")
    joblib.dump(rf, args.models_dir / "rf.joblib", compress=3)
    (args.models_dir / "labels.json").write_text(
        json.dumps({"labels": class_names, "feature_version": 2}, indent=2)
    )

    # Evaluate through the same code path the server uses.
    from predictor import Predictor

    predictor = Predictor(args.models_dir)
    X_check = v2(test_pts)
    keras_out = mlp.predict((X_check - mean) / std, verbose=0)
    numpy_out = predictor.mlp.predict_proba(X_check)
    max_diff = float(np.abs(keras_out - numpy_out).max())
    assert max_diff < 1e-4, f"NumPy MLP disagrees with Keras by {max_diff}"

    def accuracy(backend, X, y):
        if backend == "baseline":
            return accuracy_score(y, baseline.predict(X))
        return accuracy_score(y, predictor.labels[predictor.predict_proba(X, backend).argmax(1)])

    results = {}
    for name, pts in test_sets.items():
        results[name] = {"baseline": round(accuracy("baseline", v1(pts), y_test), 4)}
        for backend in ("rf", "mlp", "ensemble"):
            results[name][backend] = round(accuracy(backend, v2(pts), y_test), 4)

    # Cache keys round features to a grid; check that doesn't change answers.
    step = 0.005
    X_q = quantize(X_check, step).astype(np.float32) * step
    p_raw = predictor.predict_proba(X_check).argmax(1)
    p_q = predictor.predict_proba(X_q).argmax(1)
    quantization_agreement = round(float((p_raw == p_q).mean()), 4)

    baseline.n_jobs = 1
    latency = {
        "baseline_rf": single_sample_latency_ms(lambda x: baseline.predict_proba(x), v1(test_pts)),
        "rf": single_sample_latency_ms(lambda x: predictor.predict_proba(x, "rf"), X_check),
        "mlp_numpy": single_sample_latency_ms(lambda x: predictor.predict_proba(x, "mlp"), X_check),
        "mlp_keras": single_sample_latency_ms(lambda x: mlp(x, training=False), X_check, n=100),
        "ensemble": single_sample_latency_ms(lambda x: predictor.predict_proba(x, "ensemble"), X_check),
    }

    metrics = {
        "samples": len(points),
        "classes": class_names,
        "train_samples": len(train_pts),
        "train_samples_augmented": len(X_train),
        "test_samples": len(test_pts),
        "seed": args.seed,
        "accuracy": results,
        "cache_quantization_step": step,
        "cache_quantization_agreement": quantization_agreement,
        "single_sample_latency_ms": latency,
        "keras_numpy_max_abs_diff": max_diff,
        "per_class_report_ensemble": classification_report(
            y_test, predictor.labels[predictor.predict_proba(X_check).argmax(1)],
            output_dict=True, zero_division=0,
        ),
    }
    (args.models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nAccuracy on the held-out test set (and harder variants of it):")
    print(f"{'test set':<12}" + "".join(f"{b:>11}" for b in ("baseline", "rf", "mlp", "ensemble")))
    for name, row in results.items():
        print(f"{name:<12}" + "".join(f"{row[b] * 100:>10.2f}%" for b in ("baseline", "rf", "mlp", "ensemble")))
    print(f"\nCache quantization agreement: {quantization_agreement * 100:.2f}%")
    print("Single-sample latency (ms, median):", latency)
    print(f"\nSaved models and metrics to {args.models_dir}")


if __name__ == "__main__":
    main()
