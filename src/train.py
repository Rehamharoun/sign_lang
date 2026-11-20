"""Two-phase training: frozen MobileNetV2, then fine-tune its last 30 layers.

python src/train.py --data-dir data --num-classes 50 --out-dir models
"""
import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import VideoSequence, list_videos, load_labels, pick_sign_ids
from model import build_model, compile_model, unfreeze_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-classes", type=int, default=50)
    ap.add_argument("--classes-file", help="optional file listing the sign ids to use")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs-frozen", type=int, default=15)
    ap.add_argument("--epochs-finetune", type=int, default=15)
    a = ap.parse_args()

    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sign_ids = pick_sign_ids(data / "general_train", a.num_classes, a.classes_file)
    class_index = {sid: i for i, sid in enumerate(sign_ids)}
    labels = load_labels(data / "KARSL-502_Labels.xlsx")
    (out / "classes.json").write_text(json.dumps(
        [{"index": i, "sign_id": s, "label": labels.get(s, str(s))} for s, i in class_index.items()],
        ensure_ascii=False, indent=2), encoding="utf-8")

    items = list_videos(data / "general_train", sign_ids)
    train_items, val_items = train_test_split(items, test_size=0.15, random_state=42,
                                              stratify=[s for _, s in items])
    test_items = list_videos(data / "general_test", sign_ids)
    print(f"classes={len(sign_ids)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    train = VideoSequence(train_items, class_index, a.batch_size, True)
    val = VideoSequence(val_items, class_index, a.batch_size, False)
    test = VideoSequence(test_items, class_index, a.batch_size, False)

    model, base = build_model(len(sign_ids))
    ckpt = ModelCheckpoint(out / "sign_model.keras", monitor="val_accuracy", save_best_only=True)
    cbs = [ckpt, EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
           ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)]

    compile_model(model, 1e-3)                       # Phase 1
    model.fit(train, validation_data=val, epochs=a.epochs_frozen, callbacks=cbs)

    unfreeze_last(base, 30)                          # Phase 2
    compile_model(model, 1e-5)
    model.fit(train, validation_data=val, epochs=a.epochs_finetune, callbacks=cbs)

    loss, acc = model.evaluate(test)
    print(f"Test accuracy: {acc * 100:.2f}%")
    (out / "metrics.json").write_text(json.dumps({"test_accuracy": acc, "test_loss": loss}, indent=2))


if __name__ == "__main__":
    main()
