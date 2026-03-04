"""
model/train_multiclass.py
--------------------------
Neural network training pipeline for the 5-class BCI EEG brain-state
classifier (focus, relax, stress, sleep, meditation).

Architecture
------------
  Input(feature_size)
  → Dense(256, relu) → Dropout(0.3)
  → Dense(128, relu) → Dropout(0.3)
  → Dense(64,  relu)
  → Dense(5, softmax)          [5-class: focus/relax/stress/sleep/meditation]

The trained model is saved to model/saved_model/bci_multiclass_model.h5 and
the fitted StandardScaler to model/saved_model/scaler_multiclass.pkl.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Resolve the project root so imports work regardless of working directory
_HERE = Path(__file__).resolve().parent          # bci-eeg-project/model/
_ROOT = _HERE.parent                             # bci-eeg-project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_eeg import generate_multiclass_eeg, BRAIN_STATES
from preprocessing.filter import extract_features

# Saved-artefact paths
SAVED_MODEL_DIR = _HERE / "saved_model"
MULTICLASS_MODEL_PATH = SAVED_MODEL_DIR / "bci_multiclass_model.h5"
MULTICLASS_SCALER_PATH = SAVED_MODEL_DIR / "scaler_multiclass.pkl"
MULTICLASS_HISTORY_PATH = SAVED_MODEL_DIR / "history_multiclass.npy"

# Human-readable class names in label order (0–4)
CLASS_NAMES = [BRAIN_STATES[i] for i in sorted(BRAIN_STATES)]


def build_multiclass_model(input_dim: int, n_classes: int = 5) -> keras.Model:
    """Build and compile the 5-class Keras Sequential neural network.

    Args:
        input_dim: Number of input features.
        n_classes: Number of output classes (default: 5).

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="bci_multiclass_classifier",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_multiclass(
    n_samples: int = 2000,
    epochs: int = 50,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[keras.Model, dict]:
    """Run the full 5-class training pipeline.

    Steps:
      1. Generate simulated multiclass EEG data.
      2. Extract frequency-band features and normalise.
      3. Split into train / test sets.
      4. Build and train the neural network.
      5. Persist the model and scaler to disk.
      6. Print per-class accuracy.

    Args:
        n_samples:    Number of EEG epochs to generate (default: 2000).
        epochs:       Training epochs (default: 50).
        batch_size:   Mini-batch size (default: 32).
        test_size:    Fraction of data reserved for validation (default: 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        model:   Trained Keras model.
        history: Training history dict.
    """
    print("\n[1/4] Generating simulated multiclass EEG data …")
    signals, labels = generate_multiclass_eeg(n_samples=n_samples)
    print(f"      signals shape: {signals.shape}  labels shape: {labels.shape}")
    for state_id, name in BRAIN_STATES.items():
        count = int((labels == state_id).sum())
        print(f"      {name:>10}: {count} samples")

    print("[2/4] Extracting features …")
    features, scaler = extract_features(signals)
    print(f"      features shape: {features.shape}")

    print("[3/4] Splitting data into train/test sets …")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    print(f"      train: {X_train.shape[0]} samples  |  test: {X_test.shape[0]} samples")

    print("[4/4] Building and training model …")
    model = build_multiclass_model(input_dim=features.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
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
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------ #
    # Per-class accuracy report                                            #
    # ------------------------------------------------------------------ #
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n=== Per-Class Accuracy Report ===")
    print(
        classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, digits=4
        )
    )

    # ------------------------------------------------------------------ #
    # Persist artefacts                                                    #
    # ------------------------------------------------------------------ #
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MULTICLASS_MODEL_PATH))
    joblib.dump(scaler, str(MULTICLASS_SCALER_PATH))
    np.save(str(MULTICLASS_HISTORY_PATH), history_obj.history)

    train_acc = history_obj.history["accuracy"][-1]
    val_acc = history_obj.history["val_accuracy"][-1]
    print(f"\n✅ Training accuracy   : {train_acc * 100:.2f}%")
    print(f"✅ Validation accuracy : {val_acc * 100:.2f}%")
    print(f"\nModel saved  → {MULTICLASS_MODEL_PATH}")
    print(f"Scaler saved → {MULTICLASS_SCALER_PATH}")

    return model, history_obj.history


if __name__ == "__main__":
    train_multiclass()
