import streamlit as st
import cv2
import numpy as np
from detect import EmotionDetector
import tempfile

st.title("😊 Live Emotion Detection")
st.markdown("Uses a CNN model to detect emotions (Happy, Sad, Neutral) from a webcam feed.")

@st.cache_resource
def load_detector():
    return EmotionDetector()

try:
    detector = load_detector()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}. Did you run 'train.py'?")
    st.stop()

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect Emotions
    results = detector.detect_emotion(frame)
    
    # Draw Results
    frame = detector.draw_results(frame, results)
    
    # Convert to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
else:
    st.write("Click 'Start Webcam' to begin.")
    cap.release()
