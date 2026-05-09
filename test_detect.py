import cv2
import numpy as np
from detect import EmotionDetector
import os

def test_inference():
    print("Initializing EmotionDetector...")
    try:
        detector = EmotionDetector()
    except Exception as e:
        print(f"FAILED to load detector: {e}")
        return

    # Create a dummy image that mimics a face (roughly) to test pipeline
    # Note: Haar Cascade needs a real-ish face features or it won't detect anything.
    # Since we can't easily generate a "haar-detectable" face synthetically without complex drawing,
    # we might bypass the face_cascade for this unit test and test the prediction logic directly
    # OR we can try to use one of the generated synthetic images and see if it picks up.
    # However, synthetic images are just circles, Haar Cascade for *frontalface_default* likely won't detect them.
    
    # Let's verify the model prediction mechanics directly first.
    print("Testing model prediction logic directly...")
    
    # Create a dummy 48x48 image (Happy - Yellow Circle)
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.circle(img, (24, 24), 20, (0, 255, 255), -1) 
    
    # Preprocess exactly like detect.py
    roi = img.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)

    try:
        prediction = detector.model.predict(roi)
        class_idx = np.argmax(prediction)
        # We know classes are [Happy, Sad, Neutral]
        classes = ["Happy", "Sad", "Neutral"]
        predicted_label = classes[class_idx]
        confidence = float(prediction[0][class_idx])
        
        print(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
        print("SUCCESS: Model inference ran without error.")
        
    except Exception as e:
        print(f"FAILED during prediction: {e}")

if __name__ == "__main__":
    test_inference()
