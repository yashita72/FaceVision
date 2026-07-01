import cv2
import numpy as np
from recognizer import FaceRecognizer

# Load recognizer
recognizer = FaceRecognizer(
    "models/face_recognition_sface_2021dec.onnx"
)

database = recognizer.database

# -------------------------------------------------
# Collect all embeddings
# -------------------------------------------------
embeddings = []
labels = []

for person, emb_list in database.items():

    for i, emb in enumerate(emb_list):

        embeddings.append(emb)
        labels.append(f"{person}_{i+1}")

# -------------------------------------------------
# Print Embeddings
# -------------------------------------------------
print("=" * 100)
print("FACE EMBEDDINGS")
print("=" * 100)

for label, emb in zip(labels, embeddings):

    emb = emb.flatten()

    print(f"\n{label}")
    print("-" * 70)

    print(f"Shape : {emb.shape}")
    print(f"Min   : {emb.min():.4f}")
    print(f"Max   : {emb.max():.4f}")
    print(f"Mean  : {emb.mean():.4f}")
    print(f"Std   : {emb.std():.4f}")

    np.set_printoptions(
        precision=4,
        suppress=True,
        linewidth=120
    )

    print(emb)

# -------------------------------------------------
# Similarity Matrix
# -------------------------------------------------
print("\n\n")
print("=" * 100)
print("COSINE SIMILARITY MATRIX")
print("=" * 100)

import pandas as pd

matrix = []

for i in range(len(embeddings)):

    row = []

    for j in range(len(embeddings)):

        score = recognizer.recognizer.match(
            embeddings[i],
            embeddings[j],
            cv2.FaceRecognizerSF_FR_COSINE
        )

        row.append(round(score, 3))

    matrix.append(row)

df = pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
)

print("\n")
print(df)
print("\nDone!")