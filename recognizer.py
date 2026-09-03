import os
import cv2
import numpy as np
import pickle


class FaceRecognizer:

    def __init__(self, model_path, db_path="embeddings/embeddings.pkl"):

        self.recognizer = cv2.FaceRecognizerSF.create(
            model_path,
            ""
        )
        self.db_path = db_path
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.database = pickle.load(f)
        else:
            self.database = {}

    def align_face(self, image, face):

        return self.recognizer.alignCrop(image, face)

    def get_embedding(self, aligned_face):

        return self.recognizer.feature(aligned_face)

    def match(self, embedding, threshold=0.65):

        best_score = -1
        best_name = "Unknown"

        for person, embeddings in self.database.items():

            for saved_embedding in embeddings:

                score = self.recognizer.match(
                    embedding,
                    saved_embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:

                    best_score = score
                    best_name = person

        if best_score < threshold:
            best_name = "Unknown"

        return best_name, best_score