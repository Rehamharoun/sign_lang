"""Extract MediaPipe Holistic keypoints from KArSL-502 videos and save them as .npy arrays.

Per frame: pose (33 x [x,y,z,visibility] = 132) + left hand (21x3 = 63) + right hand (21x3 = 63) = 258 features.
(Face landmarks are intentionally excluded to match N_KEYPOINTS = 258.)

Usage:
    python src/extract_keypoints.py --data-dir /path/to/KArSL-502 --out-dir features
"""
import argparse
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

MAX_FRAMES = 30
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def collect_videos(root: Path) -> pd.DataFrame:
    rows = []
    for vid in root.rglob("*"):
        if vid.suffix.lower() not in VIDEO_EXT:
            continue
        nums = re.findall(r"\d+", vid.stem.split("_(")[0])
        rows.append({"path": str(vid), "sign_id": str(int(nums[-1])) if nums else "0"})
    return pd.DataFrame(rows)


def extract_keypoints(results) -> np.ndarray:
    def flat(lm, dims, n, vis=False):
        if lm is None:
            return np.zeros(n * dims)
        return np.array([[p.x, p.y, p.z, p.visibility] if vis else [p.x, p.y, p.z]
                         for p in lm.landmark]).flatten()

    return np.concatenate([
        flat(results.pose_landmarks, 4, 33, vis=True),
        flat(results.left_hand_landmarks, 3, 21),
        flat(results.right_hand_landmarks, 3, 21),
    ])


def process_video(path: str, holistic) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extract_keypoints(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    if not frames:
        raise ValueError("no frames read")
    if len(frames) >= MAX_FRAMES:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)]
    else:
        frames += [np.zeros_like(frames[0])] * (MAX_FRAMES - len(frames))
    return np.array(frames)


def build_split(df, encoder, holistic):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            X.append(process_video(row["path"], holistic))
            y.append(encoder.transform([row["sign_id"]])[0])
        except Exception as e:  # keep going on corrupt videos
            print("skip", row["path"], e)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing general_train/ and general_test/")
    ap.add_argument("--out-dir", default="features")
    a = ap.parse_args()

    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_train, df_test = collect_videos(data / "general_train"), collect_videos(data / "general_test")

    encoder = LabelEncoder().fit(sorted(df_train["sign_id"].unique(), key=int))
    print("classes:", len(encoder.classes_))

    with mp.solutions.holistic.Holistic(model_complexity=1, smooth_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as holistic:
        for name, df in [("train", df_train), ("test", df_test)]:
            X, y = build_split(df, encoder, holistic)
            np.save(out / f"X_{name}.npy", X)
            np.save(out / f"y_{name}.npy", y)
            print(name, X.shape, y.shape)


if __name__ == "__main__":
    main()
