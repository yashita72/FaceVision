import cv2
import gradio as gr
import numpy as np
import os
from detector import FaceDetector
from recognizer import FaceRecognizer

try:
    import spaces
    gpu_decorator = spaces.GPU
except (ImportError, Exception):
    def gpu_decorator(func):
        return func

# -----------------------------
# Load Models
# -----------------------------
MODEL_DETECTOR_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
MODEL_RECOGNIZER_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")

detector = FaceDetector(MODEL_DETECTOR_PATH)
recognizer = FaceRecognizer(MODEL_RECOGNIZER_PATH)


@gpu_decorator
def recognize_faces(image, threshold=0.65):
    """
    Process input image (RGB numpy array from Gradio):
    1. Convert RGB -> BGR
    2. Detect faces using FaceDetector (YuNet)
    3. For each face, align and extract embedding using FaceRecognizer (SFace)
    4. Match embedding with database and draw bounding box + name + score with cv2 (green if known, red if Unknown)
    5. Convert BGR -> RGB and return annotated image + summary text
    """
    if image is None:
        return None, "Please upload an image or capture one with your webcam."

    # Convert RGB (Gradio format) to BGR (OpenCV format)
    bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr = bgr_frame.copy()

    # Detect faces
    faces = detector.detect(bgr_frame)

    if faces is None or len(faces) == 0:
        return image, "No faces detected in the image."

    num_faces = len(faces)
    summary_lines = [f"### Detection & Recognition Summary\n**Total Faces Detected:** {num_faces}\n"]

    for idx, face in enumerate(faces, start=1):
        # Bounding Box coordinates
        x, y, w, h = face[:4].astype(int)

        # 5 Facial Landmarks
        landmarks = face[4:14].reshape(5, 2).astype(int)

        # Align Face
        aligned_face = recognizer.align_face(bgr_frame, face)

        # Generate Embedding
        embedding = recognizer.get_embedding(aligned_face)

        # Match with Database
        name, similarity = recognizer.match(embedding, threshold=threshold)

        # Green if known, Red if Unknown (in BGR: Green=(0, 255, 0), Red=(0, 0, 255))
        if name == "Unknown":
            color = (0, 0, 255)
            status_icon = "❓"
        else:
            color = (0, 255, 0)
            status_icon = "✅"

        # Draw Bounding Box
        cv2.rectangle(
            annotated_bgr,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Draw Landmarks
        for point in landmarks:
            cv2.circle(
                annotated_bgr,
                tuple(point),
                3,
                (255, 0, 0),
                -1
            )

        # Display Name above bounding box
        label_y = max(25, y - 35)
        cv2.putText(
            annotated_bgr,
            name,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # Display Recognition Score
        score_y = max(45, y - 10)
        cv2.putText(
            annotated_bgr,
            f"{similarity * 100:.1f}%",
            (x, score_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        summary_lines.append(
            f"{idx}. {status_icon} **Name:** {name} | **Similarity Score:** `{similarity * 100:.1f}%` (Threshold: `{threshold * 100:.1f}%`)"
        )

    # Convert back BGR to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    text_summary = "\n".join(summary_lines)

    return annotated_rgb, text_summary


# -----------------------------
# Gradio Interface Setup
# -----------------------------
with gr.Blocks(title="FaceVision - Face Detection & Recognition") as demo:
    gr.Markdown(
        """
        # 👤 FaceVision
        ### Real-Time Face Detection & Recognition with OpenCV YuNet & SFace
        Upload a picture or take a snapshot with your webcam to identify registered faces and detect unknown visitors.
        """
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                sources=["upload", "webcam"],
                type="numpy",
                label="Input Image"
            )
            threshold_slider = gr.Slider(
                minimum=0.30,
                maximum=0.95,
                value=0.65,
                step=0.01,
                label="Similarity Threshold",
                info="Cosine similarity threshold for face verification (default: 0.65)"
            )
            submit_btn = gr.Button("🔍 Run Face Recognition", variant="primary")

        with gr.Column():
            output_img = gr.Image(
                type="numpy",
                label="Annotated Image"
            )
            output_text = gr.Markdown(
                label="Summary",
                value="*Recognition results will appear here.*"
            )

    submit_btn.click(
        fn=recognize_faces,
        inputs=[input_img, threshold_slider],
        outputs=[output_img, output_text],
        api_name=False
    )

    # Example images if present in repository
    examples = []
    ex1 = os.path.join("images", "samridhi", "img1.jpeg")
    ex2 = os.path.join("images", "samridhi", "WhatsApp Image 2026-06-30 at 11.48.05 AM.jpeg")
    ex3 = os.path.join("images", "yashita", "WhatsApp Image 2026-06-30 at 11.48.04 AM.jpeg")
    if os.path.exists(ex1):
        examples.append([ex1, 0.65])
    if os.path.exists(ex2):
        examples.append([ex2, 0.65])
    if os.path.exists(ex3):
        examples.append([ex3, 0.65])

    if examples:
        gr.Examples(
            examples=examples,
            inputs=[input_img, threshold_slider],
            outputs=[output_img, output_text],
            fn=recognize_faces,
            cache_examples=False,
            label="Sample Test Images",
            api_name=False
        )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
