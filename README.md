
# Live Emotion Sentiment Analysis

A real-time AI-powered system that detects human emotions from live webcam input and performs sentiment analysis on text/messages. This project combines **Computer Vision**, **Deep Learning**, and **Natural Language Processing (NLP)** to analyze human emotions and sentiments in real time.


## Features

- Real-time face detection using webcam
- Emotion recognition from facial expressions
- Sentiment analysis on text/chat messages
- Detects emotions such as:
  - Happy
  - Sad
  - Angry
  - Neutral
  - Surprise
  - Fear
- Live prediction output
- User-friendly interface
- Deep learning model integration

<img width="636" height="547" alt="Screenshot 2026-02-08 164635" src="https://github.com/user-attachments/assets/0269ef9a-f323-46d5-8a0f-4dd19cbdcd28" />
<img width="642" height="534" alt="Screenshot 2026-02-08 164620" src="https://github.com/user-attachments/assets/8312fd79-3097-4cd1-9c2a-c28e51391a07" />
<img width="714" height="554" alt="Screenshot 2026-02-08 164930" src="https://github.com/user-attachments/assets/8ce07c75-0cea-4d1d-8b08-36e6305e97c5" />



## Tech Stack

### Programming Language
- Python

### Libraries/Frameworks
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Flask / FastAPI (if used)
- NLTK / TextBlob (for sentiment analysis)

## Project Structure

```bash
Live-Emotion-Sentiment-Analysis/
│
├── models/
│   ├── emotion_model.h5
│
├── dataset/
│
├── app.py
├── emotion_detection.py
├── sentiment_analysis.py
├── requirements.txt
├── README.md
└── static/
````

## How It Works

### Emotion Detection

* Captures live webcam feed
* Detects faces using OpenCV Haar Cascade/Deep Learning model
* Extracts facial features
* Predicts emotion label using trained CNN model

### Sentiment Analysis

* Takes user text/chat input
* Preprocesses text
* Uses NLP model to classify sentiment:

  * Positive
  * Negative
  * Neutral

## Installation

Clone repository:

```bash
git clone https://github.com/Adarshthakur-850/Live-Emotion-Sentiment-Analysis.git
```

Move into project directory:

```bash
cd Live-Emotion-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Project

```bash
python app.py
```

For FastAPI:

```bash
uvicorn app:app --reload
```

## Dataset

Common datasets used:

* FER-2013 (Facial Emotion Recognition)
* IMDB Sentiment Dataset
* Custom text dataset

## Future Improvements

* Deploy using Docker
* Add cloud deployment
* Improve model accuracy
* Add voice sentiment detection
* Mobile app integration
* Real-time dashboard monitoring

## Applications

* Mental health monitoring
* Customer feedback analysis
* Smart interview systems
* Education platforms
* Human behavior analysis
* Security systems

## Screenshots

Add project screenshots here.

## Author

**Adarsh Thakur**

GitHub: [Adarshthakur-850](https://github.com/Adarshthakur-850?utm_source=chatgpt.com)

## License

This project is licensed under the MIT License.









