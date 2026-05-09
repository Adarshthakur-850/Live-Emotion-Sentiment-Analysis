# Live Emotion & Sentiment Analysis System

A real-time facial emotion recognition system using a Convolutional Neural Network (CNN) and OpenCV.

## Features
- **CNN Classification**: Detects "Happy", "Sad", "Neutral".
- **Real-time Inference**: Processes webcam feed instantly.
- **Streamlit UI**: User-friendly interface with live video.

## Project Structure
```
Live Emotion & Sentiment Analysis/
│
├── data/generate_data.py # Synthetic data generation
├── model.py              # CNN Architecture
├── train.py              # Training script
├── detect.py             # Inference logic
├── app.py                # Streamlit UI
└── requirements.txt
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Data** (Required since we use synthetic data for demo):
    ```bash
    python data/generate_data.py
    ```

3.  **Train Model**:
    ```bash
    python train.py
    ```

## Running the Application

**Start the UI**:
```bash
streamlit run app.py
```
Click "Start Webcam" in the browser to begin detection.

## How it Works
1.  **Face Detection**: Uses Haar Cascades to locate faces.
2.  **Preprocessing**: Crops the face, resizes to 48x48, and noramlizes pixel values.
3.  **CNN Prediction**: The trained model predicts the emotion class.
