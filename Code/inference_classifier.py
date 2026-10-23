"""
Standalone OpenCV demo: shows the webcam with the detected hand, its
bounding box and the predicted letter. Press Q or Esc to quit.

Usage: python inference_classifier.py [--backend ensemble|rf|mlp]
"""

import argparse
import time

import cv2

from features import normalized_features
from hand_detector import HandDetector, draw_hand
from predictor import Predictor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", default="ensemble", choices=["rf", "mlp", "ensemble"])
parser.add_argument("--camera", default=0, type=int)
args = parser.parse_args()

predictor = Predictor(backend=args.backend)
detector = HandDetector(video=True)

# Initialize the webcam.
cap = cv2.VideoCapture(args.camera)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Stream Dimensions: {width}x{height}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    H, W, _ = frame.shape
    start = time.perf_counter()
    detection = detector.detect(frame, timestamp_ms=time.monotonic() * 1000)

    if detection is not None:
        result = predictor.predict(normalized_features(detection.points))
        elapsed_ms = (time.perf_counter() - start) * 1000

        draw_hand(frame, detection.points)

        # Bounding box around the hand, padded by 10 pixels on each side.
        xs, ys = detection.points[:, 0], detection.points[:, 1]
        x1, y1 = int(xs.min() * W) - 10, int(ys.min() * H) - 10
        x2, y2 = int(xs.max() * W) + 10, int(ys.max() * H) + 10

        confident = result["confidence"] >= 0.8
        colour = (0, 200, 0) if confident else (0, 0, 220)
        text = f"{result['label']} {result['confidence'] * 100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(frame, text, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3, cv2.LINE_AA)
        cv2.putText(frame, f"{elapsed_ms:.1f} ms", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
detector.close()
