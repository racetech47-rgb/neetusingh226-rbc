"""
model/finetune.py
-----------------
Fine-tune the pre-trained BCI model on PhysioNet EEG Motor Imagery data.

Strategy:
  1. Load pre-trained bci_model.h5.
  2. Freeze the first two layers.
  3. Replace the output layer for the new task (3 classes: rest / left / right).
  4. Train with a lower learning rate (0.0001) and early stopping.
  5. Save the fine-tuned model to model/saved_model/bci_finetuned.h5.

Usage
-----
    python main.py --mode finetune

    # or directly:
    from model.finetune import finetune_on_physionet
    finetune_on_physionet(subjects=[1, 2, 3])
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

_HERE = Path(__file__).resolve().parent        # bci-eeg-project/model/
_ROOT = _HERE.parent                           # bci-eeg-project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.train import MODEL_PATH, SAVED_MODEL_DIR

FINETUNED_MODEL_PATH = SAVED_MODEL_DIR / "bci_finetuned.h5"
N_FINETUNE_CLASSES = 3   # rest, left_hand, right_hand


def finetune_on_physionet(
    subjects: List[int] = None,
    data_dir: str = "datasets/physionet/",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.0001,
) -> None:
    """Fine-tune the pre-trained BCI model on PhysioNet Motor Imagery data.

    Args:
        subjects:      List of subject IDs to use for fine-tuning.
        data_dir:      Path to PhysioNet data directory.
        epochs:        Maximum training epochs (default: 30).
        batch_size:    Mini-batch size (default: 16).
        learning_rate: Optimiser learning rate (default: 0.0001).
    """
    if subjects is None:
        subjects = [1, 2, 3]

    # ------------------------------------------------------------------
    # Load PhysioNet data
    # ------------------------------------------------------------------
    from datasets.physionet_loader import load_epochs, extract_physionet_features

    all_X, all_y = [], []
    for sid in subjects:
        try:
            epochs_obj = load_epochs(sid, data_dir=data_dir)
            X, y = extract_physionet_features(epochs_obj)
            all_X.append(X)
            all_y.append(y)
        except Exception as exc:
            print(f"[finetune] Warning: could not load subject {sid}: {exc}")

    if not all_X:
        raise RuntimeError(
            "No PhysioNet data loaded. Run `python main.py --mode download-data` first."
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[finetune] Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

    # ------------------------------------------------------------------
    # Load pre-trained model
    # ------------------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found at {MODEL_PATH}. "
            "Run `python main.py --mode train` first."
        )

    base_model = keras.models.load_model(str(MODEL_PATH))
    print("[finetune] Pre-trained model loaded.")

    # Evaluate before fine-tuning (on first 200 PhysioNet samples)
    sample = X_val[:200]
    pre_loss, pre_acc = base_model.evaluate(
        sample,
        (y_val[:200] > 0).astype(np.int32),   # binary for original model
        verbose=0,
    )
    print(f"[finetune] Pre fine-tune accuracy (binary, val subset): {pre_acc * 100:.1f}%")

    # ------------------------------------------------------------------
    # Build fine-tuned model
    # ------------------------------------------------------------------
    # Freeze the first two layers of the base model
    for layer in base_model.layers[:2]:
        layer.trainable = False

    # Remove the old output layer and attach a new one for 3 classes
    feature_output = base_model.layers[-2].output  # last Dense before sigmoid
    new_output = layers.Dense(N_FINETUNE_CLASSES, activation="softmax", name="ft_output")(
        feature_output
    )
    ft_model = keras.Model(inputs=base_model.input, outputs=new_output)

    ft_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ft_model.summary()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    ft_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Evaluate after fine-tuning
    # ------------------------------------------------------------------
    _, post_acc = ft_model.evaluate(X_val, y_val, verbose=0)
    print(f"[finetune] Post-fine-tune accuracy (3-class, val): {post_acc * 100:.1f}%")
    print(f"[finetune] Pre accuracy: {pre_acc * 100:.1f}%  →  "
          f"Post accuracy: {post_acc * 100:.1f}%")

    # ------------------------------------------------------------------
    # Save fine-tuned model
    # ------------------------------------------------------------------
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ft_model.save(str(FINETUNED_MODEL_PATH))
    print(f"[finetune] Fine-tuned model saved → {FINETUNED_MODEL_PATH}")


if __name__ == "__main__":
    finetune_on_physionet()
