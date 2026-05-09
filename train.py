import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import build_model
import sys

# Add data generation to path if needed (though we just run it separately)

DATA_DIR = "data/emotions"
MODEL_PATH = "model.h5"
CLASSES = ["Happy", "Sad", "Neutral"]
IMG_SIZE = 48

def load_data():
    data = []
    labels = []
    
    for idx, label in enumerate(CLASSES):
        path = os.path.join(DATA_DIR, label)
        if not os.path.exists(path):
            continue
            
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(idx)
            except Exception as e:
                pass
                
    X = np.array(data).astype('float32') / 255.0
    y = to_categorical(labels, num_classes=len(CLASSES))
    return X, y

def train():
    print("Loading data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found. Please run 'python data/generate_data.py' first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples...")
    model = build_model(len(CLASSES))
    
    history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=16)
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save plot
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig('training_plot.png')
    print("Plot saved to training_plot.png")

if __name__ == "__main__":
    train()
