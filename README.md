# ASL Interpretation and Learning Tool

## Overview
The **ASL Interpretation and Learning Tool** is an assistive software solution aimed at bridging the communication gap between American Sign Language (ASL) users and non-ASL users. This project uses **computer vision** and **machine learning** for real-time ASL sign recognition and provides tools to help users learn ASL interactively.

---

## Features
- **Real-Time ASL Recognition**: Translates ASL signs captured via webcam into text using machine learning models.
- **Interactive Learning**: Displays reference images and feedback to help users learn ASL signs.
- **Intuitive User Interface**: Developed with **Streamlit**, providing real-time feedback.
- **Robust Classification Model**: Uses a **Random Forest Classifier** for sign prediction.
- **Hand Landmark Detection**: Utilizes **Google MediaPipe** for precise hand tracking.

---

## System Architecture
1. **Data Capture Layer**: Uses webcams to capture ASL gestures.
2. **Processing Layer**:
   - **Pre-Processing**: OpenCV enhances input video streams.
   - **Hand Landmark Detection**: MediaPipe extracts 21 key hand points.
   - **Classification**: Random Forest predicts the letter based on hand landmarks.
3. **User Interface Layer**: Streamlit-based web interface offers live feedback and interactive learning.

---

## Tools and Technologies
- **OpenCV**: Handles video capture and processing.
- **MediaPipe**: Detects hand landmarks.
- **Scikit-learn**: Trains the Random Forest classifier.
- **Streamlit**: Powers the web-based user interface.
- **Python**: Core programming language.

---

## Folder Structure
```
ASL_Interpretation_And_Learning_Tool/
│
├── Code/                # Main project code
│   ├── app.py           # Streamlit app entry point
│   ├── collect_imgs.py  # Script to collect ASL images
│   ├── create_dataset.py  # Prepares dataset from collected images
│   ├── inference_classifier.py  # Real-time prediction using Random Forest
│   ├── server.py        # HTTP server for serving JSON data
│   ├── train_classifier.py  # Trains the Random Forest model
│   ├── gauge_graph.html  # HTML visualization
│   ├── data.pickle      # Pickled dataset
│   ├── model.p          # Serialized trained model
│   ├── signImages/      # ASL sign images for reference
│   │   ├── A.png
│   │   ├── B.png
│   │   ├── ...
│   │   └── Z.png
│   └── state/           # State management
│       ├── __init__.py  # Package initialization
│       └── state.py     # Manages application state
│
├── ASL Interpretation And Learning Tool.pptx  # Project presentation
├── FinalReport.pdf      # Project final report
├── requirements.txt     # Dependencies
└── README.md            # Documentation (this file)
```

## How to Run the Project

### 1. Clone the repository:
```bash
git clone https://github.com/pathakbaibhav/ASL-Interpretation-And-Learning-Tool.git
cd ASL_Interpretation_And_Learning_Tool
```
### 2. Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
```
### 3. Install dependencies:
```bash
Copy code
pip install -r requirements.txt
```
### 4. Run the application:
```bash
Copy code
streamlit run Code/app.py
```
### 5. Run the server (optional):
```bash
Copy code
python Code/server.py
```