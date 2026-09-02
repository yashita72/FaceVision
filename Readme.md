# FaceVision

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/YOUR_USERNAME/FaceVision)

> 🚀 **Live Demo:** Try out the interactive web demo on [Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/FaceVision) *(Replace `YOUR_USERNAME` with your Hugging Face username)*.

A real-time face detection and recognition system built using **OpenCV**, **YuNet**, **SFace**, and **Gradio**.

---

## Features

- **Interactive Web UI**: Gradio web interface supporting both image uploads and live webcam snapshots
- **Real-time Face Detection**: OpenCV YuNet ONNX model for high-accuracy face localization
- **Facial Landmark Detection**: Detects 5 essential facial keypoints (eyes, nose, mouth corners)
- **Face Alignment**: OpenCV SFace alignment for improved recognition invariant to pose
- **Face Recognition**: SFace 128-D embedding extraction with Cosine Similarity matching
- **Unknown Face Detection**: Automated thresholding to flag unregistered individuals
- **Real-time Webcam Support**: Standalone script for live video stream recognition with FPS overlay
- **Modular Architecture**: Clean separation between detector, recognizer, and interfaces

---

## Project Pipeline

```
Image Upload / Webcam
        │
        ▼
Frame Capture / Preprocessing (RGB → BGR)
        │
        ▼
YuNet Face Detection
        │
        ▼
5 Facial Landmarks
        │
        ▼
Face Alignment & Cropping
        │
        ▼
SFace Feature Extraction (128-D Embeddings)
        │
        ▼
Cosine Similarity Matching vs Database
        │
        ▼
Annotation (Bounding Box + Landmark Dots + Name + Score)
        │
        ▼
Gradio Web Output / OpenCV Display
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
│   ├── person1/
│   └── person2/
│
├── embeddings/
│   └── embeddings.pkl
│
├── app.py                  # Gradio Web Interface (Hugging Face Spaces entrypoint)
├── webcam_demo.py          # Real-time local webcam recognition loop
├── detector.py             # FaceDetector class (YuNet)
├── recognizer.py           # FaceRecognizer class (SFace)
├── generate_embeddings.py  # Script to generate face embeddings database
├── compare_embeddings.py   # Embedding distance / similarity comparison script
├── visualize_embeddings.py # 2D/3D embedding visualization
├── utils.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python 3.x
- OpenCV & OpenCV Contrib (YuNet & SFace modules)
- Gradio (Web Application UI)
- NumPy
- Pickle

---

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/yashita72/FaceVision.git
cd FaceVision
```

### 2. Create and Activate Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Face Embeddings

Organize reference images in `images/<person_name>/` and run:

```bash
python generate_embeddings.py
```

### 5. Launch Application

- **Gradio Web Interface (Recommended / HF Spaces entrypoint):**
  ```bash
  python app.py
  ```
  Open the provided local URL (typically `http://127.0.0.1:7860`) in your web browser.

- **Real-time Webcam Demo (Local OpenCV Window):**
  ```bash
  python webcam_demo.py
  ```
  Press `q` to exit the webcam window.

---

## Current Results

- Successfully recognizes registered users with high similarity scores.
- Accurately identifies unknown individuals using cosine similarity thresholding.
- Fast, lightweight inference on standard CPU hardware without GPU requirements.
- Modular, portable structure ready for 1-click deployment on Hugging Face Spaces.

---

## Author

**Yashita Gaur**  
B.Tech AI & Data Science  
BEL AI Intern