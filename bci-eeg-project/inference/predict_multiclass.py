"""
inference/predict_multiclass.py
--------------------------------
Real-time EEG brain-state classification using the trained 5-class BCI model.

Provides:
  - predict_brain_state : classify a single EEG epoch and return the brain
                          state name with a confidence percentage.
  - demo                : generate live EEG samples and print predictions.
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

from data.simulate_eeg import generate_multiclass_eeg, BRAIN_STATES
from preprocessing.filter import extract_features
from model.train_multiclass import MULTICLASS_MODEL_PATH, MULTICLASS_SCALER_PATH

# Human-readable class names in label order (0–4)
CLASS_NAMES = [BRAIN_STATES[i].upper() for i in sorted(BRAIN_STATES)]

# Cached model / scaler — loaded once on first call
_model = None
_scaler = None


def _load_artifacts() -> None:
    """Load the saved multiclass model and scaler into module-level cache."""
    global _model, _scaler

    if _model is not None and _scaler is not None:
        return  # already loaded

    if not MULTICLASS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved multiclass model not found at {MULTICLASS_MODEL_PATH}. "
            "Run `python main.py --mode train-multi` first."
        )
    if not MULTICLASS_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Multiclass scaler not found at {MULTICLASS_SCALER_PATH}. "
            "Run `python main.py --mode train-multi` first."
        )

    _model = keras.models.load_model(str(MULTICLASS_MODEL_PATH))
    _scaler = joblib.load(str(MULTICLASS_SCALER_PATH))


def predict_brain_state(
    eeg_sample: np.ndarray,
    fs: int = 256,
) -> Tuple[str, float]:
    """Classify a single EEG epoch into one of 5 brain states.

    Args:
        eeg_sample: numpy array of shape (n_channels, n_timepoints).
        fs:         Sampling frequency in Hz (default: 256).

    Returns:
        state:      One of "FOCUS", "RELAX", "STRESS", "SLEEP", "MEDITATION".
        confidence: Prediction confidence as a percentage (0–100).
    """
    _load_artifacts()

    # Add batch dimension → (1, n_channels, n_timepoints)
    epoch = eeg_sample[np.newaxis, ...]

    # Feature extraction with the saved scaler
    features, _ = extract_features(epoch, fs=fs, normalize=False)
    features_scaled = _scaler.transform(features)

    # Inference — shape: (1, 5)
    probs: np.ndarray = _model.predict(features_scaled, verbose=0)[0]

    best_idx: int = int(np.argmax(probs))
    state: str = CLASS_NAMES[best_idx]
    confidence: float = float(probs[best_idx]) * 100.0

    return state, confidence


def demo(n_demos: int = 5) -> None:
    """Simulate real-time prediction by classifying newly generated epochs.

    Args:
        n_demos: Number of sample predictions to print (default: 5).
    """
    print(f"\n🔬 Running multiclass real-time prediction demo ({n_demos} samples) …\n")
    signals, true_labels = generate_multiclass_eeg(n_samples=n_demos)

    for i in range(n_demos):
        state, confidence = predict_brain_state(signals[i])
        true_state = BRAIN_STATES[int(true_labels[i])].upper()
        match_icon = "✓" if state == true_state else "✗"
        print(
            f"  [{match_icon}] 🧠 Brain State: {state:>10s} "
            f"(confidence: {confidence:5.1f}%)  "
            f"[true: {true_state}]"
        )

    print("\nDemo complete.\n")


if __name__ == "__main__":
    demo()
