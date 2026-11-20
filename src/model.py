"""MobileNetV2 (per-frame features) + BiLSTM (temporal) classifier."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from data import IMG_SIZE, NUM_FRAMES


def build_model(num_classes=50, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False                                     # Phase 1: frozen backbone

    inp = layers.Input((num_frames, img_size, img_size, 3))
    x = layers.Rescaling(2.0, offset=-1.0)(inp)                # [0,1] -> [-1,1] as MobileNetV2 expects
    x = layers.TimeDistributed(base)(x)                        # frame-wise feature maps
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)              # temporal sequence learning
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="mobilenetv2_bilstm"), base


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def unfreeze_last(base, n=30):
    """Phase 2: fine-tune the last `n` layers of MobileNetV2."""
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
