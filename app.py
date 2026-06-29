import cv2
from detector import FaceDetector

# Load YuNet
detector = FaceDetector("models/face_detection_yunet_2023mar.onnx")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        break

    faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # Bounding Box
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Five Facial Landmarks
            landmarks = face[4:14].reshape(5, 2).astype(int)

            for point in landmarks:

                cv2.circle(
                    frame,
                    tuple(point),
                    3,
                    (0, 0, 255),
                    -1
                )

            confidence = face[-1]

            cv2.putText(
                frame,
                f"{confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    cv2.imshow("FaceVision", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()