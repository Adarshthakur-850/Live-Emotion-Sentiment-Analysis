import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "model.h5"
CLASSES = ["Happy", "Sad", "Neutral"]
IMG_SIZE = 48

class EmotionDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise IOError("Model not found. Please train the model first.")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        
        for (x, y, w, h) in faces:
            # ROI for CNN (Color image as trained)
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0) # Add batch dimension

            prediction = self.model.predict(roi)
            class_idx = np.argmax(prediction)
            label = CLASSES[class_idx]
            confidence = float(prediction[0][class_idx])
            
            results.append((x, y, w, h, label, confidence))
            
        return results

    def draw_results(self, frame, results):
        for (x, y, w, h, label, conf) in results:
            color = (0, 255, 0) # Green box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame
