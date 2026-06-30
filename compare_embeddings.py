import cv2
from recognizer import FaceRecognizer

# Load recognizer
recognizer = FaceRecognizer(
    "models/face_recognition_sface_2021dec.onnx"
)

# Get embeddings
emb1 = recognizer.database["yashita"][0]
emb2 = recognizer.database["samridhi"][0]

# Compare
score = recognizer.recognizer.match(
    emb1,
    emb2,
    cv2.FaceRecognizerSF_FR_COSINE
)

print("Similarity :", score)