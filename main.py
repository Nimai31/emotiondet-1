import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime, timedelta
import uuid
import os
import altair as alt
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from slack_messages import send_file_to_user, send_text_to_user, send_face_with_caption
from deep_sort_realtime.deepsort_tracker import DeepSort
from pdf_report import generate_pdf, generate_bored_students_pdf



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

emotion_colors = {
    'bored': (0, 0, 255),       # Red
    'confused': (255, 165, 0),  # Orange
    'focused': (0, 255, 0),     # Green
    'frustrated': (128, 0, 128),# Purple
    'happy': (255, 255, 0),     # Yellow
    'neutral': (200, 200, 200), # Grey
    'surprised': (0, 255, 255)  # Cyan
}

# --- Engagement Weights ---
engagement_weights = {
    'focused': 1.0,
    'happy': 0.8,
    'neutral': 0.6,
    'surprised': 0.4,
    'confused': 0.3,
    'frustrated': 0.2,
    'bored': 0.0
}

def calculate_engagement_index(emotion_counts):
    total = sum(emotion_counts.values())
    if total == 0:
        return 0.0  
    weighted_sum = sum(engagement_weights[emo] * count for emo, count in emotion_counts.items())
    return weighted_sum / total

def classify_engagement(score):
    if score >= 0.8:
        return "🟢 Highly Engaged"
    elif score >= 0.6:
        return "🟡 Moderately Engaged"
    elif score >= 0.4:
        return "🟠 At Risk"
    else:
        return "🔴 Disengaged"



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
        color = emotion_colors.get(emotion_label, (255, 255, 255))  # default: white
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
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
if 'bored_counter' not in st.session_state:
    st.session_state.bored_counter = {}  # {track_id: consecutive_bored_frames}
if 'last_notification_time' not in st.session_state:
    st.session_state.last_notification_time = datetime.min
if "notified_bored_ids" not in st.session_state:
    st.session_state.notified_bored_ids = set()
if 'bored_report_sent' not in st.session_state:
    st.session_state.bored_report_sent = False





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
    cam = cv2.VideoCapture(0) 
    frame_number = 0
    output_path = f"annotated_output_{st.session_state.session_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while st.session_state.run:
        ret, frame = cam.read()
        if not ret:
            st.error("video playback completed")
            break
        frame_number += 1  # <-- Add this line here
        frame = cv2.flip(frame, 1)
        annotated, emotions = get_emotions(frame, face_model, emotion_model, tracker)
        for emotion_data in emotions:
            face_id = emotion_data["id"]
            # --- Boredom Tracking Logic ---
            emotion = emotion_data["emotion"]

            # Count consecutive bored frames
            if emotion == "bored":
                st.session_state.bored_counter[face_id] = st.session_state.bored_counter.get(face_id, 0) + 1
            else:
                st.session_state.bored_counter[face_id] = 0

            # Threshold check
            BORED_THRESHOLD = 200  # frames

            if st.session_state.bored_counter[face_id] == BORED_THRESHOLD:
                if face_id not in st.session_state.notified_bored_ids:
                    face_path = st.session_state.saved_faces.get(face_id)
                    if face_path:
                        send_face_with_caption(
                            user_id="U08PEE90BJT",
                            face_path=face_path,
                            caption=f"😴 Student ID {face_id} has been bored for a while."
                        )
                    
                    # Mark this face ID as already notified
                    st.session_state.notified_bored_ids.add(face_id)
                    st.session_state.bored_report_sent = True


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
                "frame_number": frame_number,
                "id": face_id,
                "emotion": emotion_data["emotion"],
                "confidence": round(emotion_data["confidence"] * 100, 2),
                "face_path": face_filename
            })
        CHECK_INTERVAL = timedelta(seconds=10)
        now = datetime.now()

        if now - st.session_state.last_notification_time > CHECK_INTERVAL:
            st.session_state.last_notification_time = now
            bored_ids = [id for id, count in st.session_state.bored_counter.items() if count >= BORED_THRESHOLD]
            all_ids = list({e["id"] for e in st.session_state.emotion_log})
            
            if all_ids:
                bored_ratio = len(bored_ids) / len(all_ids)
                if bored_ratio >= 0.4:
                    msg = f"😴 {len(bored_ids)} out of {len(all_ids)} students ({bored_ratio * 100:.1f}%) seem unengaged."
                    send_text_to_user(user_id="U08PEE90BJT", message=msg)

                    pdf_path = generate_bored_students_pdf(
                        session_id=st.session_state.session_id,
                        bored_ids=bored_ids,
                        saved_faces=st.session_state.saved_faces
                    )
                    send_file_to_user(user_id="U08PEE90BJT", file_path=pdf_path, message="📝 Bored Students Summary Report")

        cv2.putText(annotated, f"Frame: {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out_writer.write(annotated)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_rgb)
    cam.release()
    out_writer.release()
    st.success(f"📼 Annotated video saved")
    with open(output_path, "rb") as video_file:
        st.download_button(
            label="📥 Download Annotated Video",
            data=video_file,
            file_name=output_path,
            mime="video/mp4"
        )




# --- After Stop: Save & Analyze ---
if not st.session_state.run and st.session_state.emotion_log and not st.session_state.report_generated:
    df = pd.DataFrame(st.session_state.emotion_log)

    # --- Compute Engagement Index ---
    emotion_counts = df['emotion'].value_counts().to_dict()
    engagement_index = calculate_engagement_index(emotion_counts)
    engagement_label = classify_engagement(engagement_index)

    st.subheader("📊 Engagement Summary")
    st.metric("Engagement Index", f"{engagement_index:.2f}", help="Weighted average of emotion scores")
    st.markdown(f"**Engagement Level:** {engagement_label}")

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
        


