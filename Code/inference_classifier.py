import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model from the pickle file.
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam.
cap = cv2.VideoCapture(0)

# Retrieve the properties of the video stream.
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Stream Dimensions: {width}x{height}")

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

# Dictionary to map predictions to characters.
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}

while True:
    data_aux = []  # Auxiliary list for storing landmarks.
    x_ = []  # List to store x-coordinates.
    y_ = []  # List to store y-coordinates.

    # Capture a frame from the webcam.
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Get the frame dimensions.
    H, W, _ = frame.shape

    # Convert the frame to RGB for hand detection.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks.
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame.
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extract and normalize hand landmarks.
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate the bounding box around the hand.
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make a prediction using the pre-trained model.
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Display the predicted character on the frame.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame, predicted_character, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
        )

    # Display the frame in a window.
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
