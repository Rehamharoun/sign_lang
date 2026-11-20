"""Streamlit app: upload an Arabic sign-language video -> Arabic text + confidence + top-5 + speech.

streamlit run app/app.py
"""
import json
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from gtts import gTTS
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data import sample_frames  # noqa: E402

MODEL_PATH = ROOT / "models" / "sign_model.keras"
CLASSES_PATH = ROOT / "models" / "classes.json"

st.set_page_config(page_title="Arabic Sign Language Translator", page_icon="🤟", layout="centered")


@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return model, [c["label"] for c in sorted(classes, key=lambda c: c["index"])]


def landmark_preview(video_path):
    """Draw MediaPipe hand landmarks (21 points per hand) on a frame from the middle of the video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        res = hands.process(rgb)
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(rgb, lm, mp.solutions.hands.HAND_CONNECTIONS)
    return rgb


st.title("🤟 Arabic Sign Language Translator")
st.caption("Upload a short MP4 of a sign — get the Arabic word as text and speech.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Train it first: `python src/train.py --data-dir data` "
             "or place `sign_model.keras` and `classes.json` in `models/`.")
    st.stop()

model, labels = load_assets()
up = st.file_uploader("Upload a sign language video", type=["mp4"])

if up is not None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(up.read())
        path = tmp.name
    st.video(path)

    if st.button("Translate", type="primary"):
        try:
            with st.spinner("Processing video..."):
                preview = landmark_preview(path)
                probs = model.predict(sample_frames(path)[None, ...], verbose=0)[0]
        except Exception as e:
            st.error(f"Could not process this video: {e}")
            st.stop()

        top5 = np.argsort(probs)[::-1][:5]
        word, conf = labels[top5[0]], float(probs[top5[0]])

        if preview is not None:
            st.image(preview, caption="MediaPipe hand landmarks", use_container_width=True)
        st.markdown(f"<h1 style='text-align:center;direction:rtl'>{word}</h1>", unsafe_allow_html=True)
        st.metric("Confidence", f"{conf * 100:.1f}%")

        st.subheader("Top-5 predictions")
        for i in top5:
            st.progress(float(probs[i]), text=f"{labels[i]} — {probs[i] * 100:.1f}%")

        try:
            buf = BytesIO()
            gTTS(text=word, lang="ar").write_to_fp(buf)
            st.audio(buf.getvalue(), format="audio/mp3")
        except Exception:
            st.warning("Speech generation needs an internet connection (gTTS).")
    # uploaded video is only kept in a temp file; nothing is stored permanently
