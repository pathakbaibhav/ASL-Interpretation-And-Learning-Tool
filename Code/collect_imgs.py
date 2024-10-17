import os
import mediapipe as mp
import cv2

# Define the directory to store the collected data.
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# List of classes (letters and space) for which images will be collected.
classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", 
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
    "V", "W", "X", "Y", "Z", "space"
]

# Number of images to collect for each class.
dataset_size = 100

# Initialize MediaPipe's hand tracking module.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure the hand tracking model.
hands = mp_hands.Hands(
    static_image_mode=True, 
    min_detection_confidence=0.3, 
    max_num_hands=1
)

# Initialize the webcam.
cap = cv2.VideoCapture(0)

# Create a directory for each class to store its images.
for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_name}')

    # Wait for the user to confirm readiness to start collection.
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access the webcam.")
            break

        # Display a message on the screen to indicate readiness.
        cv2.putText(
            frame, 'Ready? Press "Q" ! :)', 
            (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 
            (0, 255, 0), 3, cv2.LINE_AA
        )
        cv2.imshow('frame', frame)

        # Break the loop when 'Q' is pressed to start collecting images.
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect images for the current class.
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Save the captured frame to the corresponding class directory.
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)

        # Display the frame on the screen.
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        counter += 1

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
