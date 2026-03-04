"""
simulate_eeg.py
---------------
Simulated EEG signal generator for the BCI EEG classifier.

Generates multi-channel EEG data for brain states:
  Binary (2-class):
    - Relax (label 0): dominant alpha waves (8–13 Hz)
    - Focus (label 1): dominant beta  waves (13–30 Hz)

  Multiclass (5-class):
    - Focus      (label 0): dominant beta (13–30 Hz)
    - Relax      (label 1): dominant alpha (8–13 Hz)
    - Stress     (label 2): high beta (25–40 Hz) + gamma (30–100 Hz)
    - Sleep      (label 3): delta (0.5–4 Hz) + theta (4–8 Hz)
    - Meditation (label 4): theta (4–8 Hz) + alpha (8–13 Hz)

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


# ------------------------------------------------------------------ #
# State name mapping for the 5-class problem                         #
# ------------------------------------------------------------------ #
BRAIN_STATES = {
    0: "focus",
    1: "relax",
    2: "stress",
    3: "sleep",
    4: "meditation",
}


def _make_epoch(
    n_channels: int,
    n_timepoints: int,
    t: np.ndarray,
    components: list,
    noise_std: float = 0.3,
) -> np.ndarray:
    """Generate a single multi-channel EEG epoch from sinusoidal components.

    Args:
        n_channels:   Number of EEG channels.
        n_timepoints: Number of time samples per channel.
        t:            Time vector of length n_timepoints.
        components:   List of (freq_low, freq_high, amp_low, amp_high) tuples
                      describing sinusoidal components to sum for each channel.
        noise_std:    Standard deviation of additive Gaussian noise.

    Returns:
        epoch: numpy array of shape (n_channels, n_timepoints).
    """
    epoch = np.zeros((n_channels, n_timepoints))
    for ch in range(n_channels):
        for (f_low, f_high, a_low, a_high) in components:
            freq = np.random.uniform(f_low, f_high)
            amp = np.random.uniform(a_low, a_high)
            epoch[ch] += amp * np.sin(2 * np.pi * freq * t)
        epoch[ch] += np.random.normal(0, noise_std, n_timepoints)
    return epoch


def generate_multiclass_eeg(
    n_samples: int = 2000,
    n_channels: int = 8,
    duration: float = 2.0,
    fs: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simulated EEG signals for 5 brain states.

    States and their dominant frequency components:
      0 — focus     : beta (13–30 Hz)
      1 — relax     : alpha (8–13 Hz)
      2 — stress    : high beta (25–40 Hz) + gamma (30–100 Hz)
      3 — sleep     : delta (0.5–4 Hz) + theta (4–8 Hz)
      4 — meditation: theta (4–8 Hz) + alpha (8–13 Hz)

    Args:
        n_samples:  Total number of EEG epochs (split evenly across 5 classes).
        n_channels: Number of EEG channels (default: 8).
        duration:   Duration of each epoch in seconds (default: 2 s).
        fs:         Sampling frequency in Hz (default: 256 Hz).

    Returns:
        signals: numpy array of shape (n_samples, n_channels, n_timepoints).
        labels:  numpy array of shape (n_samples,) with integer labels 0–4.
    """
    n_timepoints: int = int(duration * fs)
    t: np.ndarray = np.linspace(0, duration, n_timepoints, endpoint=False)

    # Per-class sinusoidal component specs:
    # Each entry is (freq_low, freq_high, amp_low, amp_high)
    state_components = {
        0: [  # focus — dominant beta
            (13.0, 30.0, 1.5, 3.0),   # primary beta
            (8.0, 13.0, 0.2, 0.5),    # weak alpha
            (4.0, 8.0, 0.1, 0.3),     # very weak theta
        ],
        1: [  # relax — dominant alpha
            (8.0, 13.0, 1.5, 3.0),    # primary alpha
            (4.0, 8.0, 0.3, 0.7),     # weak theta
            (13.0, 30.0, 0.1, 0.4),   # very weak beta
        ],
        2: [  # stress — high beta + gamma
            (25.0, 40.0, 1.5, 3.0),   # primary high beta
            (30.0, 60.0, 0.8, 1.5),   # gamma component
            (13.0, 25.0, 0.3, 0.7),   # moderate beta
        ],
        3: [  # sleep — delta + theta
            (0.5, 4.0, 2.0, 4.0),     # primary delta
            (4.0, 8.0, 1.0, 2.0),     # theta
            (8.0, 13.0, 0.1, 0.3),    # very weak alpha
        ],
        4: [  # meditation — theta + alpha
            (4.0, 8.0, 1.5, 3.0),     # primary theta
            (8.0, 13.0, 1.2, 2.5),    # alpha
            (13.0, 20.0, 0.1, 0.3),   # very weak beta
        ],
    }

    n_per_class = n_samples // 5
    remainder = n_samples - n_per_class * 5

    signals: list = []
    labels: list = []

    for state_id in range(5):
        count = n_per_class + (1 if state_id < remainder else 0)
        for _ in range(count):
            epoch = _make_epoch(
                n_channels,
                n_timepoints,
                t,
                state_components[state_id],
            )
            signals.append(epoch)
            labels.append(state_id)

    signals_arr = np.array(signals, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(len(labels_arr))

    return signals_arr[idx], labels_arr[idx]


if __name__ == "__main__":
    # Binary classifier demo
    signals, labels = generate_eeg_data(n_samples=100)
    print(f"[Binary] Signals shape : {signals.shape}")
    print(f"[Binary] Labels shape  : {labels.shape}")
    print(f"[Binary] Class counts  : relax={int((labels == 0).sum())}  focus={int((labels == 1).sum())}")

    print()

    # Multiclass demo
    mc_signals, mc_labels = generate_multiclass_eeg(n_samples=500)
    print(f"[Multi]  Signals shape : {mc_signals.shape}")
    print(f"[Multi]  Labels shape  : {mc_labels.shape}")
    for state_id, state_name in BRAIN_STATES.items():
        count = int((mc_labels == state_id).sum())
        print(f"[Multi]  {state_name:>10}: {count} samples")
