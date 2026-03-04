"""
inference/predict.py
--------------------
Real-time EEG brain-state classification using the trained BCI model.

Provides:
  - predict_state: classify a single EEG epoch and return the brain state
                   label with a confidence percentage.
  - demo         : generate a live EEG sample and print the prediction.
"""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

import tensorflow as tf
from tensorflow import keras

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_eeg import generate_eeg_data
from preprocessing.filter import extract_features
from model.train import MODEL_PATH, SCALER_PATH

# Cached model / scaler — loaded once on first call
_model = None
_scaler = None


def _load_artifacts() -> None:
    """Load the saved model and scaler into module-level cache."""
    global _model, _scaler

    if _model is not None and _scaler is not None:
        return  # already loaded

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_PATH}. "
            "Run `python main.py --mode train` first."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run `python main.py --mode train` first."
        )

    _model = keras.models.load_model(str(MODEL_PATH))
    _scaler = joblib.load(str(SCALER_PATH))


def predict_state(
    eeg_sample: np.ndarray,
    fs: int = 256,
) -> Tuple[str, float]:
    """Classify a single EEG epoch as FOCUS or RELAX.

    Args:
        eeg_sample: numpy array of shape (n_channels, n_timepoints).
        fs:         Sampling frequency in Hz (default: 256).

    Returns:
        state:      "FOCUS" or "RELAX".
        confidence: Prediction confidence in percent (0–100).
    """
    _load_artifacts()

    # Add batch dimension → (1, n_channels, n_timepoints)
    epoch = eeg_sample[np.newaxis, ...]

    # Feature extraction with the saved scaler
    features, _ = extract_features(epoch, fs=fs, normalize=False)
    features_scaled = _scaler.transform(features)

    # Inference
    prob: float = float(_model.predict(features_scaled, verbose=0).flatten()[0])

    if prob >= 0.5:
        state = "FOCUS"
        confidence = prob * 100.0
    else:
        state = "RELAX"
        confidence = (1.0 - prob) * 100.0

    return state, confidence


def demo(n_demos: int = 5) -> None:
    """Simulate real-time prediction by classifying newly generated epochs.

    Args:
        n_demos: Number of sample predictions to print (default: 5).
    """
    print(f"\n🔬 Running real-time prediction demo ({n_demos} samples) …\n")
    signals, true_labels = generate_eeg_data(n_samples=n_demos)

    for i in range(n_demos):
        state, confidence = predict_state(signals[i])
        true_label = "FOCUS" if true_labels[i] == 1 else "RELAX"
        match_icon = "✓" if state == true_label else "✗"
        print(
            f"  [{match_icon}] 🧠 Brain State: {state:5s} "
            f"(confidence: {confidence:5.1f}%)  "
            f"[true: {true_label}]"
        )

    print("\nDemo complete.\n")


if __name__ == "__main__":
    demo()
