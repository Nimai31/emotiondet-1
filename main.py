import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import uuid
import os
import altair as alt
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from slack_messages import send_file_to_user
from deep_sort_realtime.deepsort_tracker import DeepSort
from pdf_report import generate_pdf



# --- Load Models ---
@st.cache_resource
def load_models():
    face_detector = YOLO("yolov11n-face.pt")
    emotion_model = YOLO("best.pt")
    tracker = DeepSort(max_age=30)
    return face_detector, emotion_model, tracker

class_map = {
    0: 'bored',
    1: 'confused',
    2: 'focused',
    3: 'frustrated',
    4: 'happy',
    5: 'neutral',
    6: 'surprised'
}

def get_emotions(frame, face_model, emotion_model, tracker):
    results = face_model(frame)[0]
    annotated_frame = frame.copy()
    detected_emotions = []

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))  # (x, y, w, h), confidence, class

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(w), int(h)
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            continue

        face_resized = cv2.resize(face_crop, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        emotion_pred = emotion_model(face_rgb, verbose=False)[0]
        emotion_cls = int(emotion_pred.probs.top1)
        confidence = float(emotion_pred.probs.top1conf)
        emotion_label = class_map[emotion_cls]

        detected_emotions.append({
            "id": track_id,
            "emotion": emotion_label,
            "confidence": confidence,
            "face_crop": face_crop
        })

        # Draw rectangle and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID {track_id}: {emotion_label} ({confidence*100:.1f}%)"
        cv2.putText(annotated_frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated_frame, detected_emotions



# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("🎥 Real-time Emotion Detection with YOLO")
os.makedirs("faces", exist_ok=True)


face_model, emotion_model, tracker = load_models()
FRAME_WINDOW = st.image([])

if 'run' not in st.session_state:
    st.session_state.run = False
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'saved_faces' not in st.session_state:
    st.session_state.saved_faces = {}  # Track ID to filepath



col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Video"):
        st.session_state.run = True
        st.session_state.report_generated = False
with col2:
    if st.button("⏹️ Stop Video"):
        st.session_state.run = False

# --- Real-time Capture ---
if st.session_state.run:
    cam = cv2.VideoCapture('classroom.mp4')
    while st.session_state.run:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)
        annotated, emotions = get_emotions(frame, face_model, emotion_model, tracker)
        for emotion_data in emotions:
            face_id = emotion_data["id"]

            # Save face image only once per ID
            if face_id not in st.session_state.saved_faces:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                face_filename = f"faces/face_{face_id}_{timestamp}.jpg"
                cv2.imwrite(face_filename, emotion_data["face_crop"])
                st.session_state.saved_faces[face_id] = face_filename
            else:
                face_filename = st.session_state.saved_faces[face_id]

            st.session_state.emotion_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id": face_id,
                "emotion": emotion_data["emotion"],
                "confidence": round(emotion_data["confidence"] * 100, 2),
                "face_path": face_filename
            })

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_rgb)
    cam.release()

# --- After Stop: Save & Analyze ---
# --- After Stop: Save & Analyze ---
if not st.session_state.run and st.session_state.emotion_log and not st.session_state.report_generated:
    df = pd.DataFrame(st.session_state.emotion_log)
    session_id = st.session_state.session_id
    csv_file = f"emotion_log_{session_id}.csv"
    df.to_csv(csv_file, index=False)

    if os.path.exists(csv_file):
        st.success(f"📄 Emotion log saved as `{csv_file}`")

        st.info("📄 Generating per-person detailed report...")

        pdf_path = generate_pdf(session_id, df, st.session_state.saved_faces)


        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", data=f, file_name=pdf_path, mime="application/pdf")

        # Slack sending
        user_id = "U08PEE90BJT"
        send_file_to_user(user_id, pdf_path, message="Emotion Analytics Report is ready!")

        st.success("✅ Report successfully generated and sent!")
        st.success("✅ Report created successfully!")
        


