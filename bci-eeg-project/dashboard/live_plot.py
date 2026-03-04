"""
dashboard/live_plot.py
-----------------------
Real-time BCI EEG live dashboard using matplotlib FuncAnimation.

Displays three subplots updated every 200 ms:
  1. Raw EEG signal   — last 2 seconds across all channels
  2. FFT Power Spectrum — mean power in theta/alpha/beta/gamma bands
  3. Brain State Probability Bar Chart — live softmax confidence per class

Hardware modes:
  - use_real_hardware=False (default): reads simulated EEG epochs
  - use_real_hardware=True           : tries Muse then OpenBCI; falls back if
                                       neither is available

Entry point:
  start_dashboard(use_real_hardware=False)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # interactive backend; may fall back to Qt5Agg/etc.
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Resolve project root for sibling-package imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_eeg import BRAIN_STATES, generate_multiclass_eeg
from preprocessing.filter import BANDS

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #
FS: int = 256                    # Sampling frequency (Hz)
EPOCH_DURATION: float = 2.0     # Epoch length (seconds)
UPDATE_INTERVAL_MS: int = 200   # Animation update period

N_CHANNELS: int = 8
N_TIMEPOINTS: int = int(FS * EPOCH_DURATION)

# Extended bands including gamma for the spectrum plot
DISPLAY_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}
BAND_COLORS = ["#5B9BD5", "#ED7D31", "#A9D18E", "#FF6B6B"]

CLASS_NAMES = [BRAIN_STATES[i].capitalize() for i in sorted(BRAIN_STATES)]
CLASS_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

# ------------------------------------------------------------------ #
# State shared between animation frames                               #
# ------------------------------------------------------------------ #
_current_epoch: Optional[np.ndarray] = None    # (n_channels, n_timepoints)
_current_probs: Optional[np.ndarray] = None    # (5,)
_use_hardware: bool = False


def _fetch_epoch() -> np.ndarray:
    """Acquire one EEG epoch from hardware or simulation.

    Returns:
        epoch: numpy array of shape (n_channels, n_timepoints).
    """
    if _use_hardware:
        # Try Muse first, then OpenBCI — both fall back to simulation
        try:
            from hardware.muse_reader import read_epoch as muse_read
            return muse_read(duration=EPOCH_DURATION)
        except Exception:  # noqa: BLE001
            pass
        try:
            from hardware.openbci_reader import read_epoch as openbci_read
            return openbci_read(duration=EPOCH_DURATION)
        except Exception:  # noqa: BLE001
            pass

    # Simulated fallback
    signals, _ = generate_multiclass_eeg(
        n_samples=1, n_channels=N_CHANNELS, duration=EPOCH_DURATION, fs=FS
    )
    return signals[0]


def _compute_band_powers(epoch: np.ndarray) -> np.ndarray:
    """Compute mean FFT power per display band averaged over all channels.

    Args:
        epoch: numpy array (n_channels, n_timepoints).

    Returns:
        powers: numpy array of shape (n_bands,) — one value per display band.
    """
    n_timepoints = epoch.shape[1]
    freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / FS)
    powers = np.zeros(len(DISPLAY_BANDS))

    for ch_signal in epoch:
        fft_vals = np.abs(np.fft.rfft(ch_signal)) ** 2
        for b_idx, (low, high) in enumerate(DISPLAY_BANDS.values()):
            mask = (freqs >= low) & (freqs <= high)
            if mask.sum() > 0:
                powers[b_idx] += np.mean(fft_vals[mask])

    # Average across channels
    powers /= epoch.shape[0]
    return powers


def _predict_probs(epoch: np.ndarray) -> np.ndarray:
    """Run multiclass inference on *epoch* and return class probabilities.

    Falls back to uniform probabilities if the model is not loaded.

    Args:
        epoch: numpy array (n_channels, n_timepoints).

    Returns:
        probs: numpy array of shape (5,) summing to 1.
    """
    try:
        from inference.predict_multiclass import predict_brain_state, CLASS_NAMES as _CN
        from preprocessing.filter import extract_features
        import joblib
        from model.train_multiclass import MULTICLASS_MODEL_PATH, MULTICLASS_SCALER_PATH
        from tensorflow import keras

        if (
            MULTICLASS_MODEL_PATH.exists()
            and MULTICLASS_SCALER_PATH.exists()
        ):
            model = keras.models.load_model(str(MULTICLASS_MODEL_PATH))
            scaler = joblib.load(str(MULTICLASS_SCALER_PATH))
            features, _ = extract_features(epoch[np.newaxis, ...], normalize=False)
            features_scaled = scaler.transform(features)
            probs = model.predict(features_scaled, verbose=0)[0]
            return probs
    except Exception:  # noqa: BLE001
        pass

    # Uniform fallback
    return np.ones(5) / 5.0


def start_dashboard(use_real_hardware: bool = False) -> None:
    """Launch the real-time BCI live dashboard.

    Args:
        use_real_hardware: If True, attempts to connect to Muse or OpenBCI.
                           Falls back to simulated data on failure.
    """
    global _use_hardware, _current_epoch, _current_probs
    _use_hardware = use_real_hardware

    # ------------------------------------------------------------------ #
    # Figure setup                                                         #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1E1E2E")
    fig.suptitle("🧠 BCI Live Dashboard", fontsize=16, color="white", fontweight="bold")

    # Subplot 1 — Raw EEG signal
    ax_eeg = fig.add_subplot(3, 1, 1)
    ax_eeg.set_facecolor("#2A2A3E")
    ax_eeg.set_title("Raw EEG Signal (last 2 s, all channels)", color="white", fontsize=11)
    ax_eeg.set_xlabel("Time (s)", color="#AAAACC")
    ax_eeg.set_ylabel("Amplitude (µV)", color="#AAAACC")
    ax_eeg.tick_params(colors="#AAAACC")
    for spine in ax_eeg.spines.values():
        spine.set_edgecolor("#444466")

    t_axis = np.linspace(0, EPOCH_DURATION, N_TIMEPOINTS, endpoint=False)
    eeg_lines = [
        ax_eeg.plot(t_axis, np.zeros(N_TIMEPOINTS), lw=0.8, alpha=0.85)[0]
        for _ in range(N_CHANNELS)
    ]
    channel_colors = plt.cm.Set2(np.linspace(0, 1, N_CHANNELS))
    for line, color in zip(eeg_lines, channel_colors):
        line.set_color(color)

    ax_eeg.legend(
        [f"Ch {i + 1}" for i in range(N_CHANNELS)],
        loc="upper right",
        fontsize=7,
        framealpha=0.3,
        labelcolor="white",
    )

    # Subplot 2 — FFT Power Spectrum
    ax_fft = fig.add_subplot(3, 1, 2)
    ax_fft.set_facecolor("#2A2A3E")
    ax_fft.set_title("FFT Power Spectrum (band averages)", color="white", fontsize=11)
    ax_fft.set_ylabel("Mean Power", color="#AAAACC")
    ax_fft.tick_params(colors="#AAAACC")
    for spine in ax_fft.spines.values():
        spine.set_edgecolor("#444466")

    band_names = list(DISPLAY_BANDS.keys())
    band_x = np.arange(len(band_names))
    fft_bars = ax_fft.bar(
        band_x,
        np.zeros(len(band_names)),
        color=BAND_COLORS,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_fft.set_xticks(band_x)
    ax_fft.set_xticklabels(
        [b.capitalize() for b in band_names], color="white", fontsize=10
    )

    # Subplot 3 — Brain State Probability
    ax_prob = fig.add_subplot(3, 1, 3)
    ax_prob.set_facecolor("#2A2A3E")
    ax_prob.set_title("Brain State Probability", color="white", fontsize=11)
    ax_prob.set_ylabel("Confidence", color="#AAAACC")
    ax_prob.set_ylim(0, 1)
    ax_prob.tick_params(colors="#AAAACC")
    for spine in ax_prob.spines.values():
        spine.set_edgecolor("#444466")

    state_x = np.arange(len(CLASS_NAMES))
    prob_bars = ax_prob.bar(
        state_x,
        np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES),
        color=CLASS_COLORS,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_prob.set_xticks(state_x)
    ax_prob.set_xticklabels(CLASS_NAMES, color="white", fontsize=10)

    # Probability text labels above each bar
    prob_texts = [
        ax_prob.text(
            x, 0.02, "0%", ha="center", va="bottom",
            color="white", fontsize=9, fontweight="bold",
        )
        for x in state_x
    ]

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------------------ #
    # Animation update function                                            #
    # ------------------------------------------------------------------ #
    def _update(_frame: int):
        epoch = _fetch_epoch()

        # --- subplot 1: raw EEG ---
        for ch_idx, line in enumerate(eeg_lines):
            # Offset channels vertically for readability
            offset = ch_idx * 5.0
            line.set_ydata(epoch[ch_idx] + offset)

        eeg_min = epoch.min() - 5
        eeg_max = epoch.max() + N_CHANNELS * 5
        ax_eeg.set_ylim(eeg_min, eeg_max)

        # --- subplot 2: FFT ---
        powers = _compute_band_powers(epoch)
        for bar, val in zip(fft_bars, powers):
            bar.set_height(val)
        ax_fft.set_ylim(0, max(powers.max() * 1.2, 1.0))

        # --- subplot 3: probabilities ---
        probs = _predict_probs(epoch)
        best_idx = int(np.argmax(probs))
        for b_idx, (bar, prob, txt) in enumerate(
            zip(prob_bars, probs, prob_texts)
        ):
            bar.set_height(float(prob))
            bar.set_alpha(1.0 if b_idx == best_idx else 0.55)
            txt.set_text(f"{prob * 100:.1f}%")
            txt.set_y(float(prob) + 0.01)

        return (*eeg_lines, *fft_bars, *prob_bars, *prob_texts)

    # ------------------------------------------------------------------ #
    # Start animation                                                      #
    # ------------------------------------------------------------------ #
    dashboard_animation = animation.FuncAnimation(  # must hold a reference to prevent GC
        fig,
        _update,
        interval=UPDATE_INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )
    _ = dashboard_animation  # referenced to prevent garbage collection

    plt.show()


if __name__ == "__main__":
    start_dashboard(use_real_hardware=False)
