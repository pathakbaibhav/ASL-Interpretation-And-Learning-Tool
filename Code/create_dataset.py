import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe components for hand tracking.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure the hand detection model.
hands = mp_hands.Hands(
    static_image_mode=True, 
    min_detection_confidence=0.3, 
    max_num_hands=1
)

# Directory containing the collected image data.
DATA_DIR = './data'

# Lists to store data and labels.
data = []
labels = []

# Iterate over all class directories.
for dir_ in os.listdir(DATA_DIR):
    # Iterate over all images in the class directory.
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Auxiliary list for storing landmarks.
        x_ = []  # List to store x-coordinates.
        y_ = []  # List to store y-coordinates.

        # Read and convert the image to RGB format.
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks.
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Extract hand landmarks.
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect all x and y coordinates.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates by subtracting the minimum x and y values.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the normalized data and corresponding label.
            data.append(data_aux)
            labels.append(dir_)

# Save the collected data and labels to a pickle file.
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
