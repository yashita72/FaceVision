import cv2
import numpy as np
import pickle


class FaceRecognizer:

    def __init__(self, model_path):

        self.recognizer = cv2.FaceRecognizerSF.create(
            model_path,
            ""
        )

        with open("embeddings/embeddings.pkl", "rb") as f:
            self.database = pickle.load(f)

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