import cv2
import numpy as np
import os

DATA_DIR = "data/emotions"
IMG_SIZE = 48
CLASSES = ["Happy", "Sad", "Neutral"]
SAMPLES_PER_CLASS = 50

def generate_data():
    print("Generating synthetic emotion data...")
    for label in CLASSES:
        path = os.path.join(DATA_DIR, label)
        os.makedirs(path, exist_ok=True)
        
        for i in range(SAMPLES_PER_CLASS):
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            # Simple geometric features to distinguish classes for the CNN
            if label == "Happy":
                # Draw a smile (Yellow)
                cv2.circle(img, (24, 24), 20, (0, 255, 255), -1) # Face
                cv2.ellipse(img, (24, 30), (10, 5), 0, 0, 180, (0, 0, 0), 2) # Smile
            elif label == "Sad":
                # Draw a frown (Blue)
                cv2.circle(img, (24, 24), 20, (255, 0, 0), -1) # Face
                cv2.ellipse(img, (24, 35), (10, 5), 0, 180, 360, (0, 0, 0), 2) # Frown
            elif label == "Neutral":
                # Draw a straight line (Gray)
                cv2.circle(img, (24, 24), 20, (128, 128, 128), -1) # Face
                cv2.line(img, (18, 32), (30, 32), (0, 0, 0), 2) # Mouth

            # Add noise
            noise = np.random.randint(0, 50, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            cv2.imwrite(f"{path}/{i}.jpg", img)

    print(f"Generated {SAMPLES_PER_CLASS} images per class in {DATA_DIR}")

if __name__ == "__main__":
    generate_data()
