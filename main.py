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

# --- Load Models ---
@st.cache_resource
def load_models():
    face_detector = YOLO("yolov11n-face.pt")
    emotion_model = YOLO("best.pt")
    return face_detector, emotion_model

class_map = {
    0: 'bored',
    1: 'confused',
    2: 'focused',
    3: 'frustrated',
    4: 'happy',
    5: 'neutral',
    6: 'surprised'
}

def get_emotions(frame, face_model, emotion_model):
    results = face_model(frame)[0]
    annotated_frame = frame.copy()
    detected_emotions = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        emotion_pred = emotion_model(face_rgb, verbose=False)[0]
        emotion_cls = int(emotion_pred.probs.top1)
        confidence = float(emotion_pred.probs.top1conf)
        emotion_label = class_map[emotion_cls]
        detected_emotions.append((emotion_label, confidence))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{emotion_label} ({confidence*100:.1f}%)"
        cv2.putText(annotated_frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated_frame, detected_emotions

def generate_pdf(session_id, df, top3, avg_conf):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    # Emotion frequency
    df['emotion'].value_counts().plot(kind='bar', ax=axs[0], color='skyblue')
    axs[0].set_title('Emotion Frequency')
    axs[0].set_xlabel('Emotion')
    axs[0].set_ylabel('Count')

    # Timeline
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    axs[1].scatter(df['timestamp'], df['emotion'].astype(str), c='green')
    axs[1].set_title('Emotion Over Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Emotion')

    # Average confidence
    avg_conf.plot(kind='bar', x='emotion', y='confidence', ax=axs[2], color='orange')
    axs[2].set_title('Average Confidence per Emotion')
    axs[2].set_xlabel('Emotion')
    axs[2].set_ylabel('Confidence')

    charts_path = f"analytics_charts_{session_id}.png"
    plt.tight_layout()
    plt.savefig(charts_path)
    plt.close()

    pdf_path = f"emotion_analytics_{session_id}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height - 40, f"Emotion Analytics Report - Session {session_id}")
    
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 80, "Top 3 Emotions:")
    y_pos = height - 100
    for i, (emo, count) in enumerate(top3.items(), 1):
        c.drawString(40, y_pos, f"{i}. {emo.capitalize()} - {count} times")
        y_pos -= 20

    c.drawImage(charts_path, 30, 100, width=540, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()

    return pdf_path

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("🎥 Real-time Emotion Detection with YOLO")

face_model, emotion_model = load_models()
FRAME_WINDOW = st.image([])

if 'run' not in st.session_state:
    st.session_state.run = False
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Video"):
        st.session_state.run = True
with col2:
    if st.button("⏹️ Stop Video"):
        st.session_state.run = False

# --- Real-time Capture ---
if st.session_state.run:
    cam = cv2.VideoCapture(0)
    while st.session_state.run:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)
        annotated, emotions = get_emotions(frame, face_model, emotion_model)
        for label, confidence in emotions:
            st.session_state.emotion_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "emotion": label,
                "confidence": round(confidence * 100, 2)
            })
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_rgb)
    cam.release()

# --- After Stop: Save & Analyze ---
# --- After Stop: Save & Analyze ---
if not st.session_state.run and st.session_state.emotion_log:
    df = pd.DataFrame(st.session_state.emotion_log)
    session_id = st.session_state.session_id
    csv_file = f"emotion_log_{session_id}.csv"
    df.to_csv(csv_file, index=False)

    if os.path.exists(csv_file):
        st.success(f"📄 Emotion log saved as `{csv_file}`")
        with open(csv_file, "rb") as f:
            st.download_button("📥 Download Emotion Log CSV", data=f, file_name=csv_file, mime="text/csv")

        st.subheader("📊 Emotion Analysis")

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Pie Chart - Emotion vs Time Proportion
        st.markdown("#### 🥧 Time Spent in Each Emotion")
        emotion_counts = df['emotion'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Insights
        st.markdown("#### 💡 Interpretation")
        total = emotion_counts.sum()
        bored_time = emotion_counts.get('bored', 0)
        confused_time = emotion_counts.get('confused', 0)
        frustrated_time = emotion_counts.get('frustrated', 0)
        happy_time = emotion_counts.get('happy', 0)
        focused_time = emotion_counts.get('focused', 0)

        if bored_time / total > 0.25:
            st.info("📌 **Student might be bored.**")

        if (confused_time + frustrated_time) / total > 0.25:
            st.warning("❓ **Student might possibly have a doubt.**")

        if (happy_time + focused_time) / total > 0.3:
            st.success("✅ **Student understands the lesson.**")

        # Bar Chart - Emotion vs Total Frames (Time)
        st.markdown("#### 📊 Emotion Duration (Frame Count)")
        fig2, ax2 = plt.subplots()
        emotion_counts.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_xlabel("Emotion")
        ax2.set_ylabel("Time Detected (frames)")
        ax2.set_title("Emotion vs Time")
        st.pyplot(fig2)

        # Average Confidence Chart
        avg_conf = df.groupby('emotion')['confidence'].mean().reset_index()

        # Top 3 Emotions Text
        top3 = emotion_counts.head(3)
        st.markdown("#### 🏆 Top 3 Detected Emotions")
        for i, (emo, count) in enumerate(top3.items(), 1):
            st.write(f"**{i}. {emo.capitalize()}** — {count} times")

        # Generate PDF with new top3 and avg_conf
        pdf_path = generate_pdf(session_id, df, top3.to_dict(), avg_conf)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", data=f, file_name=pdf_path, mime="application/pdf")
