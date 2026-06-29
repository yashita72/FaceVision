import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame
    success, frame = camera.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("FaceVision", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()