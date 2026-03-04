"""
model/evaluate.py
-----------------
Evaluation utilities for the trained BCI EEG classifier.

Loads the saved model and scaler, then:
  1. Generates a fresh held-out test set.
  2. Prints a full classification report (precision, recall, F1).
  3. Plots a colour-coded confusion matrix.
  4. Plots training history (accuracy & loss curves).

All plots are saved as PNG files inside model/saved_model/.
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in all envs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_eeg import generate_eeg_data
from preprocessing.filter import extract_features
from model.train import MODEL_PATH, SCALER_PATH, HISTORY_PATH

# Output directory for plots
PLOTS_DIR = _HERE / "saved_model"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
HISTORY_PLOT_PATH = PLOTS_DIR / "training_history.png"

CLASS_NAMES = ["Relax", "Focus"]


def _plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    """Render and save a colour-coded confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(CONFUSION_MATRIX_PATH), dpi=120)
    plt.close(fig)
    print(f"Confusion matrix saved → {CONFUSION_MATRIX_PATH}")


def _plot_history(history: dict) -> None:
    """Render and save training accuracy and loss curves."""
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy subplot
    axes[0].plot(epochs, history["accuracy"], label="Train accuracy", color="steelblue")
    axes[0].plot(epochs, history["val_accuracy"], label="Val accuracy", color="darkorange")
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss subplot
    axes[1].plot(epochs, history["loss"], label="Train loss", color="steelblue")
    axes[1].plot(epochs, history["val_loss"], label="Val loss", color="darkorange")
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(HISTORY_PLOT_PATH), dpi=120)
    plt.close(fig)
    print(f"Training history plot saved → {HISTORY_PLOT_PATH}")


def evaluate(n_test_samples: int = 200) -> None:
    """Load the saved model and evaluate it on freshly generated test data.

    Args:
        n_test_samples: Number of EEG epochs to generate for evaluation
                        (default: 200).
    """
    # ------------------------------------------------------------------ #
    # Load artefacts                                                       #
    # ------------------------------------------------------------------ #
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

    model = keras.models.load_model(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))

    # ------------------------------------------------------------------ #
    # Generate evaluation data                                            #
    # ------------------------------------------------------------------ #
    print(f"\nGenerating {n_test_samples} evaluation samples …")
    signals, labels = generate_eeg_data(n_samples=n_test_samples)
    features, _ = extract_features(signals, normalize=False)
    features = scaler.transform(features)

    # ------------------------------------------------------------------ #
    # Predictions                                                         #
    # ------------------------------------------------------------------ #
    y_prob = model.predict(features, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # ------------------------------------------------------------------ #
    # Classification report                                               #
    # ------------------------------------------------------------------ #
    print("\n=== Classification Report ===")
    print(
        classification_report(
            labels, y_pred, target_names=CLASS_NAMES, digits=4
        )
    )

    # ------------------------------------------------------------------ #
    # Confusion matrix plot                                               #
    # ------------------------------------------------------------------ #
    cm = confusion_matrix(labels, y_pred)
    _plot_confusion_matrix(cm, CLASS_NAMES)

    # ------------------------------------------------------------------ #
    # Training history plot                                               #
    # ------------------------------------------------------------------ #
    if HISTORY_PATH.exists():
        history = np.load(str(HISTORY_PATH), allow_pickle=True).item()
        _plot_history(history)
    else:
        print(f"History file not found at {HISTORY_PATH} — skipping history plot.")


if __name__ == "__main__":
    evaluate()
