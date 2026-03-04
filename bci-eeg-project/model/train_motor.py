"""
model/train_motor.py
---------------------
Train a motor imagery classifier for 4 classes:
  0 — left hand
  1 — right hand
  2 — feet
  3 — rest

Architecture
------------
  Input(feature_size)
  → Dense(512, relu) → Dropout(0.4)
  → Dense(256, relu) → Dropout(0.4)
  → Dense(128, relu) → Dropout(0.3)
  → Dense(4,  softmax)

Outputs
-------
  model/saved_model/bci_motor_model.h5
  model/saved_model/scaler_motor.pkl

Usage
-----
    python main.py --mode train-motor
"""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_motor_eeg import generate_motor_eeg
from preprocessing.filter import extract_features

SAVED_MODEL_DIR    = _HERE / "saved_model"
MOTOR_MODEL_PATH   = SAVED_MODEL_DIR / "bci_motor_model.h5"
MOTOR_SCALER_PATH  = SAVED_MODEL_DIR / "scaler_motor.pkl"

CLASS_NAMES = ["left_hand", "right_hand", "feet", "rest"]


def build_motor_model(input_dim: int) -> keras.Model:
    """Build and compile the motor imagery Keras model.

    Args:
        input_dim: Number of input features.

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(4, activation="softmax"),
        ],
        name="bci_motor_classifier",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_motor(
    n_samples: int = 2000,
    epochs: int = 60,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[keras.Model, dict]:
    """Run the motor imagery training pipeline.

    Args:
        n_samples:    EEG epochs to generate (default: 2000).
        epochs:       Training epochs (default: 60).
        batch_size:   Mini-batch size (default: 32).
        test_size:    Validation fraction (default: 0.2).
        random_state: Random seed.

    Returns:
        model:   Trained Keras model.
        history: Training history dict.
    """
    print("\n[1/4] Generating motor imagery EEG data …")
    signals, labels = generate_motor_eeg(n_samples=n_samples)
    print(f"      signals shape: {signals.shape}  labels shape: {labels.shape}")

    print("[2/4] Extracting features …")
    # Use first 8 channels for feature extraction (consistent with binary model)
    signals_8ch = signals[:, :8, :]
    features, scaler = extract_features(signals_8ch)
    print(f"      features shape: {features.shape}")

    print("[3/4] Splitting data …")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Class weights to handle any imbalance
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights_arr))

    print("[4/4] Building and training motor imagery model …")
    model = build_motor_model(input_dim=features.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Per-class accuracy
    print("\n📊 Per-class accuracy on test set:")
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = y_test == cls_id
        if mask.sum() > 0:
            acc = (y_pred[mask] == cls_id).mean()
            print(f"  {cls_name:12s}: {acc * 100:.1f}%")

    # Save artefacts
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MOTOR_MODEL_PATH))
    joblib.dump(scaler, str(MOTOR_SCALER_PATH))

    overall_acc = history_obj.history["val_accuracy"][-1]
    print(f"\n✅ Validation accuracy: {overall_acc * 100:.2f}%")
    print(f"Model saved  → {MOTOR_MODEL_PATH}")
    print(f"Scaler saved → {MOTOR_SCALER_PATH}")

    return model, history_obj.history


if __name__ == "__main__":
    train_motor()
