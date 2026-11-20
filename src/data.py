"""Shared video / data utilities (used by training and by the Streamlit app)."""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

IMG_SIZE = 128          # frames are resized to 128x128
NUM_FRAMES = 16         # frames sampled uniformly per video
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def sample_frames(video_path, num_frames=NUM_FRAMES, size=IMG_SIZE):
    """Uniformly sample `num_frames` frames, resize and normalise to [0, 1]. Returns (T, H, W, 3) float32."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(total - 1, 0), num_frames, dtype=int).tolist())
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in idx:
            frame = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
            frames.append(frame.astype("float32") / 255.0)
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"Could not read frames from {video_path}")
    while len(frames) < num_frames:                      # pad short videos
        frames.append(frames[-1])
    return np.stack(frames[:num_frames])


def sign_id_from_name(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem.split("_(")[0])
    return int(nums[-1]) if nums else -1


def list_videos(root: Path, sign_ids):
    """Return [(video_path, sign_id)] for the requested sign ids under `root`."""
    wanted = set(sign_ids)
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXT:
            sid = sign_id_from_name(p)
            if sid in wanted:
                out.append((p, sid))
    return out


def load_labels(xlsx_path: Path) -> dict:
    """SignID -> Arabic label from KARSL-502_Labels.xlsx."""
    df = pd.read_excel(xlsx_path)
    return dict(zip(df["SignID"].astype(int), df["Sign-Arabic"].astype(str)))


def pick_sign_ids(train_root: Path, num_classes: int, classes_file=None):
    """Use an explicit list of sign ids (one per line / JSON list) or fall back to the first N ids found."""
    if classes_file:
        text = Path(classes_file).read_text(encoding="utf-8").strip()
        ids = json.loads(text) if text.startswith("[") else [int(x) for x in text.split()]
        return [int(i) for i in ids]
    ids = sorted({sign_id_from_name(p) for p in train_root.rglob("*") if p.suffix.lower() in VIDEO_EXT})
    return [i for i in ids if i > 0][:num_classes]


class VideoSequence(Sequence):
    def __init__(self, items, class_index, batch_size=8, shuffle=True, **kw):
        super().__init__(**kw)
        self.items, self.class_index = list(items), class_index
        self.batch_size, self.shuffle = batch_size, shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.items) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.items)

    def __getitem__(self, i):
        batch = self.items[i * self.batch_size:(i + 1) * self.batch_size]
        X = np.stack([sample_frames(p) for p, _ in batch])
        y = np.array([self.class_index[s] for _, s in batch], dtype="int32")
        return X, y
