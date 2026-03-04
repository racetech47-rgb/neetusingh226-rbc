"""
data/simulate_motor_eeg.py
--------------------------
Simulated motor imagery EEG signal generator.

Four motor-imagery classes are generated using physiologically inspired
frequency-band suppression patterns:

  0 — left hand  : contralateral right-hemisphere mu/beta suppression (right channels)
  1 — right hand : contralateral left-hemisphere mu/beta suppression  (left channels)
  2 — feet       : bilateral central mu/beta suppression
  3 — rest       : baseline alpha + beta across all channels

Uses 64 channels labelled according to the standard 10-20 system.

Usage
-----
    from data.simulate_motor_eeg import generate_motor_eeg
    X, y = generate_motor_eeg(n_samples=2000)
"""

import numpy as np
from typing import Tuple, List

# -----------------------------------------------------------------------
# Standard 10-20 system channel names (64-channel layout)
# -----------------------------------------------------------------------
CHANNEL_NAMES_64: List[str] = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7",  "C3",  "Cz",  "C4",  "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7",  "P3",  "Pz",  "P4",  "P8",
    "O1",  "Oz",  "O2",
    "AF7", "AF3", "AF4", "AF8",
    "F5",  "F1",  "F2",  "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10",
    "C5",  "C1",  "C2",  "C6",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "P5",  "P1",  "P2",  "P6",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "P9",  "P10",
    "Oz2", "O9",  "O10",
]

# Channel index groups for left/right/central hemispheres
_LEFT_CH_IDX   = [0, 2, 3, 7, 11, 16, 20, 25, 32, 37, 42, 46, 51, 55]
_RIGHT_CH_IDX  = [1, 6, 9, 10, 14, 18, 19, 24, 27, 35, 40, 44, 50, 58]
_CENTRAL_CH_IDX = [4, 8, 13, 17, 22, 28, 43, 48]


def generate_motor_eeg(
    n_samples: int = 2000,
    n_channels: int = 64,
    duration: float = 4.0,
    fs: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simulated motor imagery EEG data for 4 classes.

    Args:
        n_samples:   Total number of EEG epochs (split evenly across 4 classes).
        n_channels:  Number of EEG channels (default: 64).
        duration:    Epoch duration in seconds (default: 4 s).
        fs:          Sampling frequency in Hz (default: 250 Hz).

    Returns:
        X: numpy array of shape (n_samples, n_channels, n_timepoints).
        y: numpy array of shape (n_samples,) with integer labels 0-3.
    """
    n_timepoints = int(duration * fs)
    t = np.linspace(0, duration, n_timepoints, endpoint=False)

    per_class = n_samples // 4
    remainder = n_samples - per_class * 4

    signals: list = []
    labels: list  = []

    for cls in range(4):
        count = per_class + (1 if cls < remainder else 0)
        for _ in range(count):
            epoch = _generate_epoch(cls, n_channels, t, fs)
            signals.append(epoch)
            labels.append(cls)

    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _generate_epoch(
    cls: int,
    n_channels: int,
    t: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Generate a single EEG epoch for the given class.

    Args:
        cls:        Class label (0=left, 1=right, 2=feet, 3=rest).
        n_channels: Number of channels.
        t:          Time vector.
        fs:         Sampling frequency.

    Returns:
        epoch: numpy array of shape (n_channels, len(t)).
    """
    n_timepoints = len(t)
    epoch = np.zeros((n_channels, n_timepoints), dtype=np.float32)

    # Clamp channel index lists to the actual channel count
    left_idx    = [i for i in _LEFT_CH_IDX   if i < n_channels]
    right_idx   = [i for i in _RIGHT_CH_IDX  if i < n_channels]
    central_idx = [i for i in _CENTRAL_CH_IDX if i < n_channels]

    for ch in range(n_channels):
        # Baseline: alpha + beta across all channels
        alpha_freq = np.random.uniform(8.0,  13.0)
        beta_freq  = np.random.uniform(13.0, 30.0)
        epoch[ch] += 1.0 * np.sin(2 * np.pi * alpha_freq * t)
        epoch[ch] += 0.5 * np.sin(2 * np.pi * beta_freq  * t)
        epoch[ch] += np.random.normal(0, 0.3, n_timepoints)

    # Event-related (de)synchronisation (ERD) per class
    if cls == 0:
        # Left hand: ERD in right hemisphere (mu/beta suppression)
        _apply_erd(epoch, right_idx, t, suppression=0.7)

    elif cls == 1:
        # Right hand: ERD in left hemisphere
        _apply_erd(epoch, left_idx, t, suppression=0.7)

    elif cls == 2:
        # Feet: bilateral central ERD
        _apply_erd(epoch, central_idx, t, suppression=0.65)

    # cls == 3 (rest): unchanged baseline

    return epoch


def _apply_erd(
    epoch: np.ndarray,
    channel_indices: list,
    t: np.ndarray,
    suppression: float = 0.6,
) -> None:
    """Apply event-related desynchronisation (amplitude suppression) in-place.

    Reduces mu/beta power on the specified channels by *suppression* factor.

    Args:
        epoch:           EEG epoch (n_channels, n_timepoints) — modified in-place.
        channel_indices: List of channel indices to suppress.
        t:               Time vector.
        suppression:     Fraction of mu/beta amplitude to subtract (0–1).
    """
    n_timepoints = len(t)
    for ch in channel_indices:
        if ch >= epoch.shape[0]:
            continue
        # Generate the mu/beta component being suppressed
        mu_freq   = np.random.uniform(8.0, 12.0)
        beta_freq = np.random.uniform(13.0, 30.0)
        mu_signal = (
            suppression * np.sin(2 * np.pi * mu_freq   * t)
            + suppression * 0.5 * np.sin(2 * np.pi * beta_freq * t)
        )
        epoch[ch] -= mu_signal


if __name__ == "__main__":
    X, y = generate_motor_eeg(n_samples=400)
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        names = ["left_hand", "right_hand", "feet", "rest"]
        print(f"  class {u} ({names[u]}): {c} samples")
