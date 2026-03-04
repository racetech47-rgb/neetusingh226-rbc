"""
tests/test_preprocessing.py
-----------------------------
Tests for EEG signal preprocessing utilities.

Covers:
  - bandpass_filter()  → output shape and dtype
  - extract_features() → feature vector size
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.filter import bandpass_filter, extract_features, BANDS


N_CHANNELS    = 8
N_TIMEPOINTS  = 512
FS            = 256


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    """Tests for preprocessing.filter.bandpass_filter."""

    def test_1d_output_shape(self):
        signal = np.random.randn(N_TIMEPOINTS).astype(np.float32)
        out = bandpass_filter(signal, lowcut=8.0, highcut=13.0, fs=FS)
        assert out.shape == signal.shape

    def test_2d_output_shape(self):
        data = np.random.randn(N_CHANNELS, N_TIMEPOINTS).astype(np.float32)
        out = bandpass_filter(data, lowcut=13.0, highcut=30.0, fs=FS)
        assert out.shape == data.shape

    def test_output_dtype_preserved(self):
        signal = np.random.randn(N_TIMEPOINTS)
        out = bandpass_filter(signal, lowcut=4.0, highcut=8.0, fs=FS)
        # Output should be floating point
        assert np.issubdtype(out.dtype, np.floating)

    def test_filters_attenuate_out_of_band(self):
        """After alpha bandpass, beta-band power should be lower."""
        t = np.linspace(0, 2.0, N_TIMEPOINTS, endpoint=False)
        # Pure beta signal
        signal = np.sin(2 * np.pi * 20.0 * t)
        filtered = bandpass_filter(signal, lowcut=8.0, highcut=13.0, fs=FS)
        # After filtering, remaining power should be small
        power_ratio = np.var(filtered) / np.var(signal)
        assert power_ratio < 0.5, f"Bandpass did not attenuate: ratio={power_ratio:.3f}"


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    """Tests for preprocessing.filter.extract_features."""

    def _make_eeg(self, n_samples: int = 20) -> np.ndarray:
        return np.random.randn(n_samples, N_CHANNELS, N_TIMEPOINTS).astype(np.float32)

    def test_feature_vector_size(self):
        eeg = self._make_eeg(20)
        features, _ = extract_features(eeg, fs=FS)
        expected_dim = N_CHANNELS * len(BANDS)   # 8 channels × 3 bands = 24
        assert features.shape == (20, expected_dim), (
            f"Expected ({20}, {expected_dim}), got {features.shape}"
        )

    def test_normalised_zero_mean(self):
        """With normalize=True the feature matrix should have ~0 mean."""
        eeg = self._make_eeg(200)
        features, _ = extract_features(eeg, fs=FS, normalize=True)
        col_means = np.abs(features.mean(axis=0))
        assert np.all(col_means < 0.1), f"Features not zero-mean: max={col_means.max():.4f}"

    def test_scaler_returned(self):
        from sklearn.preprocessing import StandardScaler
        eeg = self._make_eeg(10)
        _, scaler = extract_features(eeg, fs=FS)
        assert isinstance(scaler, StandardScaler)

    def test_no_nan_or_inf(self):
        eeg = self._make_eeg(20)
        features, _ = extract_features(eeg, fs=FS)
        assert not np.any(np.isnan(features)), "NaN in features"
        assert not np.any(np.isinf(features)), "Inf in features"

    def test_output_dtype(self):
        eeg = self._make_eeg(10)
        features, _ = extract_features(eeg, fs=FS)
        assert np.issubdtype(features.dtype, np.floating)
