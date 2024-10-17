import json
import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from state import State
from time import time
from streamlit_pills import pills

# Initialize state object to maintain session state across reruns.
state = State()
state.load_state()

# Initialize variables for confidence score and detected character.
confidence = 0
predicted_character = ""

def write_value_to_file(value):
    """Writes a gauge value to a JSON file."""
    with open("gauge_value.json", "w") as f:
        json.dump({"value": value}, f)

# Load the pre-trained model for character recognition.
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam and set FPS.
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))

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

# Timer and state variables for hand detection.
detection_start_time = None
last_detected_character = ""
hold_time = 2.5  # Seconds to confirm a character detection.

# Configure the Streamlit page.
st.set_page_config(page_title="ASL", layout='wide')

# Styling for the page layout.
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h2 style='text-align: center;'>Sign Language Interpretation and Learning Tool</h2>",
    unsafe_allow_html=True
)
st.markdown('#')

# Set the default value for character selection.
value = "clear"

def clear_word():
    """Clears the current word being predicted."""
    state.clear_state()

def backspace_word():
    """Removes the last character from the current word."""
    state.set_current_word(state.get_current_word()[:-1])

# Create the main UI layout with two columns.
with st.container():
    col1, col2 = st.columns([0.6, 0.4], gap='large')

    with col1:
        # Placeholder for the video stream from the webcam.
        videoStream = st.empty()

        # Selection UI for reference characters.
        col3, col4 = st.columns([0.6, 0.4], gap="large")
        with col3:
            st.header("Select Letter For Sign Reference")
            value = pills("", ['A', 'B', 'C', 'D', 'E', 'F'], None)
        
        with col4:
            st.button("Clear", on_click=clear_word)
            st.button("Backspace", on_click=backspace_word)

    # Displays for predicted letter, progress, and the word.
    predictedLetterDisplay = st.empty()
    progress_bar = st.progress(0)
    wordDisplay = st.empty()

# Main loop to process the video stream and detect hand gestures.
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and process the video frame.
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the video stream.
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # TODO: Add model inference logic for gesture recognition.
        confidence = np.random.rand()  # Placeholder for confidence score.
        predicted_character = np.random.choice(['A', 'B', 'C', 'D'])

        # Logic for tracking detection duration and updating the word.
        if last_detected_character == predicted_character:
            if detection_start_time is None:
                detection_start_time = time()
            elif time() - detection_start_time >= hold_time and confidence > 0.8:
                if predicted_character == "space":
                    predicted_character = " "
                state.append_to_current_word(predicted_character)
                detection_start_time = None  # Reset the timer.
        else:
            last_detected_character = predicted_character
            detection_start_time = None  # Reset the timer.
    else:
        last_detected_character = ""
        confidence = 0
        write_value_to_file(confidence * 100)
        detection_start_time = None  # Reset the timer if no hands are detected.

    # Display the predicted letter and update the progress bar.
    colour = "green" if confidence >= 0.8 else "red"
    predictedLetterDisplay.header(
        f"Predicted Letter: :{colour}[{predicted_character}]"
    )

    if detection_start_time:
        elapsed_time = time() - detection_start_time
        progress_value = int((elapsed_time / hold_time) * 100)
        progress_bar.progress(progress_value if confidence > 0.8 else 0)
    else:
        progress_bar.progress(0)  # Clear the progress bar.

    wordDisplay.header(
        f"Predicted Word: :green[{state.get_current_word()}]"
    )

    # Display the video stream in the Streamlit app.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 360))
    videoStream.image(frame, channels="RGB", use_column_width="auto")

    cv2.waitKey(1)

# Release the webcam and close OpenCV windows.
cap.release()
cv2.destroyAllWindows()
