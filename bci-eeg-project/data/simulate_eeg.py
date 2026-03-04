"""
simulate_eeg.py
---------------
Simulated EEG signal generator for the BCI Focus vs Relax classifier.

Generates multi-channel EEG data for two brain states:
  - Relax (label 0): dominant alpha waves (8–13 Hz)
  - Focus (label 1): dominant beta  waves (13–30 Hz)

Realistic Gaussian noise is added to each channel.
"""

import numpy as np
from typing import Tuple


def generate_eeg_data(
    n_samples: int = 1000,
    n_channels: int = 8,
    duration: float = 2.0,
    fs: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simulated EEG signals for focus and relax states.

    Args:
        n_samples:  Total number of EEG epochs to generate (split evenly between
                    the two classes).
        n_channels: Number of EEG channels (default: 8).
        duration:   Duration of each epoch in seconds (default: 2 s).
        fs:         Sampling frequency in Hz (default: 256 Hz).

    Returns:
        signals: numpy array of shape (n_samples, n_channels, n_timepoints)
                 containing the simulated EEG epochs.
        labels:  numpy array of shape (n_samples,) with integer labels
                 0 = relax, 1 = focus.
    """
    n_timepoints: int = int(duration * fs)
    t: np.ndarray = np.linspace(0, duration, n_timepoints, endpoint=False)

    half: int = n_samples // 2
    signals: list = []
    labels: list = []

    # ------------------------------------------------------------------ #
    # Relax epochs  — dominant alpha band (8–13 Hz)                       #
    # ------------------------------------------------------------------ #
    for _ in range(half):
        epoch = np.zeros((n_channels, n_timepoints))
        for ch in range(n_channels):
            # Primary alpha component
            alpha_freq = np.random.uniform(8.0, 13.0)
            alpha_amp = np.random.uniform(1.5, 3.0)
            epoch[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

            # Weak theta component (4–8 Hz)
            theta_freq = np.random.uniform(4.0, 8.0)
            theta_amp = np.random.uniform(0.3, 0.7)
            epoch[ch] += theta_amp * np.sin(2 * np.pi * theta_freq * t)

            # Very weak beta component
            beta_freq = np.random.uniform(13.0, 30.0)
            beta_amp = np.random.uniform(0.1, 0.4)
            epoch[ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t)

            # Gaussian noise
            epoch[ch] += np.random.normal(0, 0.3, n_timepoints)

        signals.append(epoch)
        labels.append(0)  # relax

    # ------------------------------------------------------------------ #
    # Focus epochs  — dominant beta band (13–30 Hz)                       #
    # ------------------------------------------------------------------ #
    for _ in range(n_samples - half):
        epoch = np.zeros((n_channels, n_timepoints))
        for ch in range(n_channels):
            # Primary beta component
            beta_freq = np.random.uniform(13.0, 30.0)
            beta_amp = np.random.uniform(1.5, 3.0)
            epoch[ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t)

            # Weak alpha component
            alpha_freq = np.random.uniform(8.0, 13.0)
            alpha_amp = np.random.uniform(0.2, 0.5)
            epoch[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

            # Weak theta component
            theta_freq = np.random.uniform(4.0, 8.0)
            theta_amp = np.random.uniform(0.1, 0.3)
            epoch[ch] += theta_amp * np.sin(2 * np.pi * theta_freq * t)

            # Gaussian noise
            epoch[ch] += np.random.normal(0, 0.3, n_timepoints)

        signals.append(epoch)
        labels.append(1)  # focus

    # Shuffle the dataset so classes are interleaved
    signals_arr = np.array(signals, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(n_samples)

    return signals_arr[idx], labels_arr[idx]


if __name__ == "__main__":
    signals, labels = generate_eeg_data(n_samples=100)
    print(f"Signals shape : {signals.shape}")
    print(f"Labels shape  : {labels.shape}")
    print(f"Class counts  : relax={int((labels == 0).sum())}  focus={int((labels == 1).sum())}")
