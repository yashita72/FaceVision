import os
import sys

# Force CPU execution to prevent ZeroGPU CUDA shim crashes with TensorFlow & OpenCV
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import gradio as gr
import numpy as np
from detector import FaceDetector
from recognizer import FaceRecognizer
from deepface import DeepFace

# -----------------------------
# Load Models & Warmup
# -----------------------------
MODEL_DETECTOR_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
MODEL_RECOGNIZER_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")

detector = FaceDetector(MODEL_DETECTOR_PATH)
recognizer = FaceRecognizer(MODEL_RECOGNIZER_PATH)

# Warm up DeepFace models on CPU during startup to avoid first-inference latency
try:
    DeepFace.build_model("Emotion")
    DeepFace.build_model("Age")
    DeepFace.build_model("Gender")
except Exception:
    pass


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


def analyze_attributes(image):
    """
    Process input image for Emotion, Age & Gender:
    1. Convert RGB -> BGR
    2. Detect faces using FaceDetector (YuNet)
    3. Crop each face and run DeepFace.analyze(face_crop, actions=['emotion','age','gender'], enforce_detection=False)
    4. Draw bounding box and label ('Emotion | Age | Gender')
    5. Return annotated image and markdown summary
    """
    if image is None:
        return None, "Please upload an image or capture one with your webcam."

    # Convert RGB to BGR
    bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr = bgr_frame.copy()
    img_h, img_w = bgr_frame.shape[:2]

    # Detect faces using YuNet detector
    faces = detector.detect(bgr_frame)

    if faces is None or len(faces) == 0:
        return image, "No faces detected in the image."

    num_faces = len(faces)
    summary_lines = [f"### Emotion, Age & Gender Summary\n**Total Faces Detected:** {num_faces}\n"]

    for idx, face in enumerate(faces, start=1):
        x, y, w, h = face[:4].astype(int)
        landmarks = face[4:14].reshape(5, 2).astype(int)

        # Safely crop the face
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = bgr_frame[y1:y2, x1:x2]

        try:
            results = DeepFace.analyze(
                img_path=face_crop,
                actions=["emotion", "age", "gender"],
                enforce_detection=False,
                silent=True
            )
            res = results[0] if isinstance(results, list) else results

            dominant_emotion = str(res.get("dominant_emotion", "Unknown")).capitalize()
            age = int(res.get("age", 0))

            # Gender parsing
            gender = res.get("dominant_gender")
            if not gender or gender == "Unknown":
                gender_dict = res.get("gender", {})
                if isinstance(gender_dict, dict) and gender_dict:
                    gender = max(gender_dict, key=gender_dict.get)
                else:
                    gender = "Unknown"
            gender = str(gender).capitalize()

            # Emotion confidence
            dominant_raw = res.get("dominant_emotion", "")
            emotion_conf = res.get("emotion", {}).get(dominant_raw, 0.0)

            label = f"{dominant_emotion} | {age} | {gender}"
        except Exception as e:
            dominant_emotion = "Unknown"
            age = "N/A"
            gender = "Unknown"
            emotion_conf = 0.0
            label = "Analysis Error"

        # Bounding box & landmark styling (reusing cv2 style)
        color = (0, 255, 0)
        cv2.rectangle(
            annotated_bgr,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        for point in landmarks:
            cv2.circle(
                annotated_bgr,
                tuple(point),
                3,
                (255, 0, 0),
                -1
            )

        # Label above bounding box
        label_y = max(25, y - 10)
        cv2.putText(
            annotated_bgr,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        summary_lines.append(
            f"{idx}. 👤 **Face #{idx}:**\n"
            f"   - **Dominant Emotion:** {dominant_emotion} ({emotion_conf:.1f}%)\n"
            f"   - **Estimated Age:** {age} years\n"
            f"   - **Gender:** {gender}\n"
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    text_summary = "\n".join(summary_lines)

    return annotated_rgb, text_summary


# -----------------------------
# Gradio Interface Setup
# -----------------------------
with gr.Blocks(title="FaceVision - Face Recognition & Attribute Analysis") as demo:
    gr.Markdown(
        """
        # 👤 FaceVision
        ### Real-Time Face Detection, Recognition & Attribute Analysis with OpenCV YuNet, SFace & DeepFace
        """
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

    with gr.Tabs():
        with gr.Tab("🎯 Face Recognition"):
            gr.Markdown("Upload a picture or take a snapshot with your webcam to identify registered faces and detect unknown visitors.")
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

        with gr.Tab("🎭 Emotion, Age & Gender"):
            gr.Markdown("Detect facial attributes including dominant emotion, estimated age, and predicted gender per face.")
            with gr.Row():
                with gr.Column():
                    attr_input_img = gr.Image(
                        sources=["upload", "webcam"],
                        type="numpy",
                        label="Input Image"
                    )
                    attr_submit_btn = gr.Button("✨ Analyze Attributes", variant="primary")

                with gr.Column():
                    attr_output_img = gr.Image(
                        type="numpy",
                        label="Annotated Image"
                    )
                    attr_output_text = gr.Markdown(
                        label="Summary",
                        value="*Attribute analysis results will appear here.*"
                    )

            attr_submit_btn.click(
                fn=analyze_attributes,
                inputs=[attr_input_img],
                outputs=[attr_output_img, attr_output_text],
                api_name=False
            )

            attr_examples = []
            if os.path.exists(ex1):
                attr_examples.append([ex1])
            if os.path.exists(ex2):
                attr_examples.append([ex2])
            if os.path.exists(ex3):
                attr_examples.append([ex3])

            if attr_examples:
                gr.Examples(
                    examples=attr_examples,
                    inputs=[attr_input_img],
                    outputs=[attr_output_img, attr_output_text],
                    fn=analyze_attributes,
                    cache_examples=False,
                    label="Sample Test Images",
                    api_name=False
                )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
