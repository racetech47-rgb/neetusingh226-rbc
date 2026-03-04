"""
preprocessing/filter.py
------------------------
EEG signal preprocessing utilities.

Provides:
  - bandpass_filter : zero-phase Butterworth bandpass filter using SOS format.
  - extract_features: FFT-based band-power feature extraction per channel,
                      followed by StandardScaler normalisation.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# Frequency band definitions (Hz)
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int = 256,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to EEG data.

    Args:
        data:    Input signal array.  Can be 1-D (n_timepoints,) or
                 2-D (n_channels, n_timepoints).
        lowcut:  Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs:      Sampling frequency in Hz (default: 256).
        order:   Filter order (default: 4).

    Returns:
        Filtered signal with the same shape as *data*.
    """
    nyq: float = 0.5 * fs
    low: float = lowcut / nyq
    high: float = highcut / nyq

    # Clamp to valid range to prevent numerical errors
    low = max(low, 1e-6)
    high = min(high, 1.0 - 1e-6)

    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _band_power(
    signal: np.ndarray,
    fs: int,
    low: float,
    high: float,
) -> float:
    """Compute the average power of *signal* within a frequency band.

    Args:
        signal: 1-D time-series array.
        fs:     Sampling frequency in Hz.
        low:    Lower band frequency in Hz.
        high:   Upper band frequency in Hz.

    Returns:
        Mean power (float) within the requested band.
    """
    n: int = len(signal)
    freqs: np.ndarray = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals: np.ndarray = np.fft.rfft(signal)
    power: np.ndarray = np.abs(fft_vals) ** 2

    band_mask = (freqs >= low) & (freqs <= high)
    if band_mask.sum() == 0:
        return 0.0
    return float(np.mean(power[band_mask]))


def extract_features(
    eeg_data: np.ndarray,
    fs: int = 256,
    normalize: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """Extract frequency-band power features from a set of EEG epochs.

    For each epoch and each channel the function computes the mean spectral
    power in the theta, alpha, and beta bands after applying the corresponding
    bandpass filter.  The result is a 1-D feature vector per epoch of length
    n_channels × n_bands.

    Args:
        eeg_data:  numpy array of shape (n_samples, n_channels, n_timepoints).
        fs:        Sampling frequency in Hz (default: 256).
        normalize: If True (default) apply StandardScaler to the feature matrix.

    Returns:
        features: numpy array of shape (n_samples, n_channels * n_bands).
        scaler:   Fitted StandardScaler instance (or a passthrough object when
                  normalize=False).
    """
    n_samples, n_channels, _ = eeg_data.shape
    n_bands: int = len(BANDS)
    features: np.ndarray = np.zeros((n_samples, n_channels * n_bands), dtype=np.float32)

    band_items = list(BANDS.items())

    for i in range(n_samples):
        feat_idx: int = 0
        for ch in range(n_channels):
            channel_signal: np.ndarray = eeg_data[i, ch, :]
            for _, (low, high) in band_items:
                filtered = bandpass_filter(channel_signal, low, high, fs=fs)
                features[i, feat_idx] = _band_power(filtered, fs, low, high)
                feat_idx += 1

    scaler = StandardScaler()
    if normalize:
        features = scaler.fit_transform(features)
    else:
        # Fit the scaler but don't transform — callers may need the scaler
        scaler.fit(features)

    return features, scaler
