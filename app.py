import cv2
from detector import FaceDetector
from recognizer import FaceRecognizer
import time

# -----------------------------
# Load Models
# -----------------------------
detector = FaceDetector("models/face_detection_yunet_2023mar.onnx")
recognizer = FaceRecognizer("models/face_recognition_sface_2021dec.onnx")

# -----------------------------
# Open Webcam
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam")
    exit()
prev_time = time.time()
# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = camera.read()

    if not success:
        break

    faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # Bounding Box
            x, y, w, h = face[:4].astype(int)

            # Facial Landmarks
            landmarks = face[4:14].reshape(5, 2).astype(int)

            # Align Face
            aligned_face = recognizer.align_face(frame, face)

            # Generate Embedding
            embedding = recognizer.get_embedding(aligned_face)

            # Match with Database
            name, similarity = recognizer.match(embedding)

            # Green = Known
            # Red = Unknown
            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Draw Bounding Box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Draw Landmarks
            for point in landmarks:
                cv2.circle(
                    frame,
                    tuple(point),
                    3,
                    (255, 0, 0),
                    -1
                )

            # Display Name
            cv2.putText(
                frame,
                name,
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            current_time = time.time()
            fps = 1/(current_time-prev_time)
            prev_time=current_time

            # Display Recognition Score
            cv2.putText(
                frame,
              f"{similarity*100:.1f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            cv2.putText(
            frame,
             f"FPS : {int(fps)}",
             (20,40),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.8,
               (255,255,0),
                2
                )

    cv2.imshow("FaceVision", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()