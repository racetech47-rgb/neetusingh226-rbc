"""
model/train.py
--------------
Neural network definition and training pipeline for the BCI EEG classifier.

Architecture
------------
  Input(feature_size)
  → Dense(128, relu) → Dropout(0.3)
  → Dense(64,  relu) → Dropout(0.3)
  → Dense(1, sigmoid)          [binary classification: 0=relax, 1=focus]

The trained model is saved to model/saved_model/bci_model.h5 and the fitted
StandardScaler to model/saved_model/scaler.pkl.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Resolve the project root so imports work regardless of working directory
_HERE = Path(__file__).resolve().parent          # bci-eeg-project/model/
_ROOT = _HERE.parent                             # bci-eeg-project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_eeg import generate_eeg_data
from preprocessing.filter import extract_features

# Saved-artefact paths
SAVED_MODEL_DIR = _HERE / "saved_model"
MODEL_PATH = SAVED_MODEL_DIR / "bci_model.h5"
SCALER_PATH = SAVED_MODEL_DIR / "scaler.pkl"
HISTORY_PATH = SAVED_MODEL_DIR / "history.npy"


def build_model(input_dim: int) -> keras.Model:
    """Build and compile the Keras Sequential neural network.

    Args:
        input_dim: Number of input features.

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="bci_eeg_classifier",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    n_samples: int = 1000,
    epochs: int = 50,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[keras.Model, dict]:
    """Run the full training pipeline.

    Steps:
      1. Generate simulated EEG data.
      2. Extract frequency-band features and normalise.
      3. Split into train / test sets.
      4. Build and train the neural network.
      5. Persist the model and scaler to disk.

    Args:
        n_samples:    Number of EEG epochs to generate (default: 1000).
        epochs:       Training epochs (default: 50).
        batch_size:   Mini-batch size (default: 32).
        test_size:    Fraction of data reserved for validation (default: 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        model:   Trained Keras model.
        history: Training history dict (keys: loss, accuracy, val_loss,
                 val_accuracy).
    """
    print("\n[1/4] Generating simulated EEG data …")
    signals, labels = generate_eeg_data(n_samples=n_samples)
    print(f"      signals shape: {signals.shape}  labels shape: {labels.shape}")

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
    model = build_model(input_dim=features.shape[1])
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
    # Persist artefacts                                                    #
    # ------------------------------------------------------------------ #
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    joblib.dump(scaler, str(SCALER_PATH))
    np.save(str(HISTORY_PATH), history_obj.history)

    # Final metrics
    train_acc = history_obj.history["accuracy"][-1]
    val_acc = history_obj.history["val_accuracy"][-1]
    print(f"\n✅ Training accuracy   : {train_acc * 100:.2f}%")
    print(f"✅ Validation accuracy : {val_acc * 100:.2f}%")
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")

    return model, history_obj.history


if __name__ == "__main__":
    train()
