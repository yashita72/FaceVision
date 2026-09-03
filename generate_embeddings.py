import os
import cv2
import pickle
import numpy as np

from detector import FaceDetector
from recognizer import FaceRecognizer

# -----------------------------
# Load Models
# -----------------------------
detector = FaceDetector("models/face_detection_yunet_2023mar.onnx")
recognizer = FaceRecognizer("models/face_recognition_sface_2021dec.onnx")

database = {}

dataset_path = "images"

# -----------------------------
# Read every person's folder
# -----------------------------
for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    print(f"\n========== {person} ==========")

    database[person] = []

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            print(f"[!] Could not read {image_name}")
            continue

        # Resize very large images
        h, w = image.shape[:2]

        if max(h, w) > 800:
            scale = 800 / max(h, w)

            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale))
            )

        # Detect Faces
        faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            print(f"[-] No face detected : {image_name}")
            continue

        # Pick highest confidence face
        best_face = faces[np.argmax(faces[:, -1])]

        confidence = best_face[-1]

        print(f"[+] {image_name}  | Confidence = {confidence:.3f}")

        try:

            # Align face
            aligned_face = recognizer.align_face(image, best_face)

            # Generate embedding
            embedding = recognizer.get_embedding(aligned_face)

            database[person].append(embedding)

        except Exception as e:

            print(f"Embedding Error : {e}")

# -----------------------------
# Save Embeddings
# -----------------------------

os.makedirs("embeddings", exist_ok=True)

with open("embeddings/embeddings.pkl", "wb") as f:
    pickle.dump(database, f)

print("\n=================================")
print("Embeddings generated successfully!")
print("Saved to embeddings/embeddings.pkl")
print("=================================")

# Print Summary
for person in database:
    print(f"{person} : {len(database[person])} embeddings")