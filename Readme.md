# FaceVision

A real-time face detection and recognition system built using **OpenCV**, **YuNet**, and **SFace**.

## Features

- Real-time Face Detection using YuNet
- Facial Landmark Detection (5 Key Points)
- Face Alignment
- Face Recognition using SFace
- 128-D Face Embedding Generation
- Cosine Similarity Matching
- Unknown Face Detection
- Embedding Database Generation
- Real-time Webcam Recognition
- Modular Project Structure

---

## Project Pipeline

```
Webcam
   │
   ▼
Frame Capture
   │
   ▼
YuNet Face Detection
   │
   ▼
5 Facial Landmarks
   │
   ▼
Face Alignment
   │
   ▼
SFace Feature Extraction
   │
   ▼
128-D Face Embeddings
   │
   ▼
Cosine Similarity
   │
   ▼
Known / Unknown Recognition
```

---

## Folder Structure

```
FaceVision/
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── images/
│
├── embeddings/
│
├── app.py
├── detector.py
├── recognizer.py
├── generate_embeddings.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- OpenCV
- YuNet Face Detector
- SFace Face Recognizer
- NumPy
- Pickle

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/yashita72/FaceVision.git
cd FaceVision
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Face Embeddings

```bash
python generate_embeddings.py
```

### Start Face Recognition

```bash
python app.py
```

---

## Current Results

- Successfully recognizes registered users.
- Detects unknown users using cosine similarity threshold.
- Runs in real time using webcam.
- Generates face embeddings for each registered person.

---

## Future Improvements

- Automatic Face Enrollment
- Attendance System
- Emotion Detection
- Streamlit Web Interface
- Anti-Spoofing
- Mask Detection

---

## Author

**Yashita Gaur**

B.Tech AI & Data Science

BEL AI Intern