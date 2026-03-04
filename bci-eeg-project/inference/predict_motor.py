"""
inference/predict_motor.py
--------------------------
Motor imagery intent classification using the trained BCI motor model.

Maps model predictions to concrete assistive-technology actions:
  LEFT HAND  → "Move cursor left / Turn left"
  RIGHT HAND → "Move cursor right / Turn right"
  FEET       → "Move forward / Scroll down"
  REST       → "Stop / No action"

Usage
-----
    from inference.predict_motor import predict_motor_intent
    intent, confidence = predict_motor_intent(eeg_sample)

    python main.py --mode predict-motor
"""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

from tensorflow import keras

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.filter import extract_features

# Saved model / scaler paths (set by train_motor.py)
_MOTOR_MODEL_PATH  = _ROOT / "model" / "saved_model" / "bci_motor_model.h5"
_MOTOR_SCALER_PATH = _ROOT / "model" / "saved_model" / "scaler_motor.pkl"

# Class index → label
_CLASS_NAMES = ["LEFT HAND", "RIGHT HAND", "FEET", "REST"]

# Action mapping
_ACTIONS = {
    "LEFT HAND":  "Move cursor left / Turn left",
    "RIGHT HAND": "Move cursor right / Turn right",
    "FEET":       "Move forward / Scroll down",
    "REST":       "Stop / No action",
}

# Emojis per intent
_EMOJIS = {
    "LEFT HAND":  "🖐️ ←",
    "RIGHT HAND": "🖐️ →",
    "FEET":       "🦶 ↓",
    "REST":       "✋",
}

# Module-level cache
_motor_model  = None
_motor_scaler = None


def _load_motor_artifacts() -> None:
    """Load the saved motor model and scaler into module-level cache."""
    global _motor_model, _motor_scaler

    if _motor_model is not None and _motor_scaler is not None:
        return

    if not _MOTOR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Motor model not found at {_MOTOR_MODEL_PATH}. "
            "Run `python main.py --mode train-motor` first."
        )
    if not _MOTOR_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Motor scaler not found at {_MOTOR_SCALER_PATH}. "
            "Run `python main.py --mode train-motor` first."
        )

    _motor_model  = keras.models.load_model(str(_MOTOR_MODEL_PATH))
    _motor_scaler = joblib.load(str(_MOTOR_SCALER_PATH))


def predict_motor_intent(
    eeg_sample: np.ndarray,
    fs: int = 250,
) -> Tuple[str, float]:
    """Classify a single EEG epoch as one of 4 motor imagery states.

    Args:
        eeg_sample: numpy array of shape (n_channels, n_timepoints).
                    Only the first 8 channels are used.
        fs:         Sampling frequency in Hz (default: 250).

    Returns:
        intent:     One of "LEFT HAND", "RIGHT HAND", "FEET", "REST".
        confidence: Prediction confidence as a percentage (0–100).
    """
    _load_motor_artifacts()

    # Use first 8 channels (consistent with training)
    sample_8ch = eeg_sample[:8, :][np.newaxis, ...]

    features, _ = extract_features(sample_8ch, fs=fs, normalize=False)
    features_scaled = _motor_scaler.transform(features)

    probs = _motor_model.predict(features_scaled, verbose=0).flatten()
    cls_idx    = int(np.argmax(probs))
    confidence = float(probs[cls_idx]) * 100.0
    intent     = _CLASS_NAMES[cls_idx]

    return intent, confidence


def demo_motor(n_demos: int = 6) -> None:
    """Run a motor imagery prediction demo using simulated data.

    Args:
        n_demos: Number of predictions to display (default: 6).
    """
    from data.simulate_motor_eeg import generate_motor_eeg

    CLASS_LABELS = ["LEFT HAND", "RIGHT HAND", "FEET", "REST"]

    print(f"\n🤖 Motor Imagery Prediction Demo ({n_demos} samples)\n")
    signals, true_labels = generate_motor_eeg(n_samples=n_demos, n_channels=64)

    for i in range(n_demos):
        intent, confidence = predict_motor_intent(signals[i], fs=250)
        true_intent = CLASS_LABELS[true_labels[i]]
        match = "✓" if intent == true_intent else "✗"
        emoji  = _EMOJIS.get(intent, "🧠")
        action = _ACTIONS.get(intent, "")
        print(
            f"  [{match}] {emoji} Motor Intent: {intent} "
            f"(confidence: {confidence:.1f}%)  "
            f"→ {action}  [true: {true_intent}]"
        )

    print("\nDemo complete.\n")


if __name__ == "__main__":
    demo_motor()
