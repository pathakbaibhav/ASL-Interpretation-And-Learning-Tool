"""
Turns the images collected by collect_imgs.py (./data/<class>/*.jpg) into
hand landmark features and saves them to data.pickle for train_classifier.py.

Features are stored in the v1 layout (landmarks shifted to start at (0, 0))
so the file stays compatible with the original dataset; training converts
them to v2.
"""

import os
import pickle

import cv2

from features import legacy_features
from hand_detector import HandDetector

# Directory containing the collected image data.
DATA_DIR = './data'

# Lists to store data and labels.
data = []
labels = []
skipped = 0

with HandDetector(lighting="auto") as detector:
    # Iterate over all class directories.
    for dir_ in sorted(os.listdir(DATA_DIR)):
        class_dir = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(class_dir):
            continue

        # Iterate over all images in the class directory.
        for img_path in sorted(os.listdir(class_dir)):
            img = cv2.imread(os.path.join(class_dir, img_path))
            if img is None:
                continue

            detection = detector.detect(img)
            if detection is None:
                skipped += 1
                continue

            data.append(legacy_features(detection.points).tolist())
            labels.append(dir_)

print(f"Extracted {len(data)} samples; {skipped} images had no detectable hand")

# Save the collected data and labels to a pickle file.
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
