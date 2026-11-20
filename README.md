<div align="center">

# 🤟 Arabic Sign Language Translator

**Translating Arabic Sign Language videos into Arabic text and speech using Computer Vision and Deep Learning.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmarks-0097A7)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Capstone Project — **Sign-Link Innovators (TrioDigi)** · DiGiLiANS Digital Pioneers Initiative · 2026

[Live Demo](https://bit.ly/4uHNjLA) · [Project Presentation](docs/Presentation_Arabic_Sign_Language.pptx)

</div>

---

## 📌 Overview

Communication barriers between sign language users and the wider community are especially visible in healthcare,
education and public services. Most sign-language recognition research targets English/American Sign Language,
while **Arabic Sign Language remains under-served**.

This project is a web-based system that takes a **recorded sign-language video** and returns the recognised
**Arabic word as text, a confidence score, top-5 predictions, and spoken audio**.

## ✨ Features

- Upload an MP4 video of a sign and get an instant translation
- Hand landmark detection and tracking with **MediaPipe** (21 points per hand, x/y/z)
- Gesture classification with a **MobileNetV2 + BiLSTM** deep learning model (50 Arabic signs)
- Predicted word shown with confidence and **top-5** predictions
- Arabic **text-to-speech** output using gTTS
- Simple Streamlit interface; no special hardware required, and uploaded videos are not stored

## 🧠 Method

```
MP4 upload ─► Frame extraction ─► Resize 128×128 + normalise ─► MobileNetV2 (per-frame features)
          ─► BiLSTM (temporal modelling) ─► Dense + Softmax (50 classes) ─► Arabic text ─► gTTS speech
```

| Stage | Details |
|---|---|
| Spatial features | MobileNetV2 applied per frame via `TimeDistributed`, then `GlobalAveragePooling2D` + `BatchNormalization` |
| Temporal model | Bidirectional LSTM |
| Classifier | Dense (ReLU) + Dropout + Softmax over 50 classes |
| Training | Phase 1: frozen MobileNetV2 → Phase 2: fine-tune last 30 layers |
| Optimiser / loss | Adam / Sparse Categorical Crossentropy |

## 📊 Results

| Metric | Value |
|---|---|
| Test accuracy (50-sign subset) | **93.25%** |
| Inference time per video | ~5 seconds |

> **Note:** the training and app code in this repository is a clean re-implementation of the architecture described above.
> The 93.25% figure was obtained in the team's original training run; re-training with these scripts may give slightly different numbers.

## 🗂️ Dataset

The model is trained on a **50-sign subset** of **KArSL-502**, an Arabic Sign Language video dataset of 502 signs
recorded by three signers. The subset used for the final model contains roughly 6,000+ video samples
(multiple takes per sign, recorded under different conditions).

> The dataset is **not included** in this repository because of its size. Download KArSL-502 from its original source
> and place it as described below.

```
data/
├── KARSL-502_Labels.xlsx
├── general_train/<sign_id>/*.mp4
└── general_test/<sign_id>/*.mp4
```

## 📁 Repository structure

```
arabic-sign-language-translator/
├── app/app.py               # Streamlit application
├── models/                   # trained model + classes.json (created by train.py)
├── notebooks/
│   └── 01_exploration_and_baseline.ipynb   # dataset exploration + baseline CNN-LSTM
├── src/
│   ├── data.py               # frame sampling, dataset utilities
│   ├── model.py              # MobileNetV2 + BiLSTM
│   ├── train.py              # two-phase training + evaluation
│   └── extract_keypoints.py  # optional: MediaPipe Holistic keypoints (258 features / frame)
├── docs/
│   └── Presentation_Arabic_Sign_Language.pptx
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting started

```bash
git clone https://github.com/<your-username>/arabic-sign-language-translator.git
cd arabic-sign-language-translator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Train the model** (frozen MobileNetV2, then fine-tune the last 30 layers)

```bash
python src/train.py --data-dir data --num-classes 50 --out-dir models
```

**Run the app**

```bash
streamlit run app/app.py
```

**Extract MediaPipe keypoints from the dataset**

```bash
python src/extract_keypoints.py --data-dir data --out-dir features
```

**Explore / train the baseline** — set `KARSL_DIR=data` and open `notebooks/01_exploration_and_baseline.ipynb`.

## 🛠️ Tech stack

Python · TensorFlow / Keras · MediaPipe · OpenCV · NumPy · Pandas · scikit-learn · Streamlit · gTTS · Google Colab · GitHub

## 🔭 Future work

- Sentence-level and **real-time** sign recognition
- Larger vocabulary and more signers / Arabic sign dialects
- Faster inference for longer videos; cloud deployment and a mobile app
- AI avatar that generates sign language from Arabic text

## 👥 Team — Sign-Link Innovators (TrioDigi)

| Member | Contribution |
|---|---|
| **Reham Haroun Mousa** | Data collection, dataset preparation and preprocessing (cleaning, labelling, feature extraction) |
| **Engy Safwat Nagy** | Deep learning model design, training and hyper-parameter tuning |
| **Gellan Romany George** | Model evaluation, optimisation, Streamlit implementation and deployment |

**Supervisor:** Dr. Rania ElGohary · **Head of Track:** Dr. Mohamed El Badawi · **Academic team:** Eng. Engy Ayman, Eng. Sama

## 📚 References

- Camgoz et al., 2018 — *Neural Sign Language Translation*, CVPR
- Lin et al., 2023 — *Gloss-Free End-to-End Sign Language Translation*, ACL
- Sivasankar et al., 2025 — *Real-Time Bidirectional Sign Language Translator Through Machine Learning*, IJERT

## 📄 License

Released under the [MIT License](LICENSE).
