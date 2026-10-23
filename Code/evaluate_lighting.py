"""
Measures how often a hand is still found as frames get darker, with and
without the detector's low-light fallback, and writes models/lighting.json.

By default it uses the hand photos in tests/fixtures (each also mirrored
and downscaled, to give more varied samples). Point --images at a folder of
your own webcam captures (e.g. ./data from collect_imgs.py) for a more
representative measurement.

Usage: python evaluate_lighting.py [--images ./data] [--limit 300]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hand_detector import HandDetector

HERE = Path(__file__).resolve().parent
BRIGHTNESS_LEVELS = [1.0, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05]


def load_images(folder, limit):
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    images = [img for img in (cv2.imread(str(p)) for p in paths[:limit]) if img is not None]
    if folder == HERE / "tests" / "fixtures":
        images += [cv2.flip(i, 1) for i in images] + [cv2.resize(i, None, fx=0.6, fy=0.6) for i in images]
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default=HERE / "tests" / "fixtures", type=Path)
    parser.add_argument("--limit", default=300, type=int)
    parser.add_argument("--out", default=HERE / "models" / "lighting.json", type=Path)
    args = parser.parse_args()

    images = load_images(args.images, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    # Only count images where a hand is found in normal light.
    with HandDetector(lighting="off", min_confidence=0.5) as plain, \
            HandDetector(lighting="auto", min_confidence=0.5) as adaptive:
        images = [img for img in images if plain.detect(img) is not None]
        print(f"{len(images)} images with a detectable hand in normal light\n")
        print(f"{'brightness':>10} {'mean px':>8} {'plain':>8} {'fallback':>9}")

        rows = []
        for level in BRIGHTNESS_LEVELS:
            frames = [(img.astype(np.float32) * level).astype(np.uint8) for img in images]
            found_plain = sum(plain.detect(f) is not None for f in frames)
            found_adaptive = sum(adaptive.detect(f) is not None for f in frames)
            rows.append({
                "brightness_factor": level,
                "mean_pixel": round(float(np.mean([f.mean() for f in frames])), 1),
                "detected_plain": found_plain,
                "detected_with_fallback": found_adaptive,
                "images": len(frames),
            })
            print(f"{level:>10} {rows[-1]['mean_pixel']:>8} "
                  f"{found_plain / len(frames):>8.0%} {found_adaptive / len(frames):>9.0%}")

    dim = [r for r in rows if r["brightness_factor"] <= 0.25]
    plain_rate = sum(r["detected_plain"] for r in dim) / sum(r["images"] for r in dim)
    adaptive_rate = sum(r["detected_with_fallback"] for r in dim) / sum(r["images"] for r in dim)
    summary = {
        "source": str(args.images),
        "images": len(images),
        "levels": rows,
        "low_light_detection_rate_plain": round(plain_rate, 4),
        "low_light_detection_rate_with_fallback": round(adaptive_rate, 4),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nLow light (brightness <= 25%): {plain_rate:.0%} -> {adaptive_rate:.0%} of hands detected")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
