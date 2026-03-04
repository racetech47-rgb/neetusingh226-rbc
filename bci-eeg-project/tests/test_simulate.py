"""
tests/test_simulate.py
----------------------
Tests for EEG signal simulation functions.

Covers:
  - generate_eeg_data()       → binary focus/relax data
  - generate_multiclass_eeg() → 5-class data (via API simulation)
  - generate_motor_eeg()      → 4-class motor imagery data
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Add project root to sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent   # bci-eeg-project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# generate_eeg_data — binary classifier
# ---------------------------------------------------------------------------

class TestGenerateEegData:
    """Tests for data.simulate_eeg.generate_eeg_data."""

    def setup_method(self):
        from data.simulate_eeg import generate_eeg_data
        self.generate_eeg_data = generate_eeg_data

    def test_default_shape(self):
        X, y = self.generate_eeg_data(n_samples=100)
        assert X.shape == (100, 8, 512), f"Unexpected shape: {X.shape}"
        assert y.shape == (100,)

    def test_custom_channels(self):
        X, y = self.generate_eeg_data(n_samples=50, n_channels=16)
        assert X.shape[1] == 16

    def test_label_values(self):
        _, y = self.generate_eeg_data(n_samples=200)
        assert set(np.unique(y)).issubset({0, 1}), "Labels should be 0 or 1 only"

    def test_both_classes_present(self):
        _, y = self.generate_eeg_data(n_samples=100)
        assert 0 in y and 1 in y, "Both classes should be present"

    def test_dtype(self):
        X, y = self.generate_eeg_data(n_samples=10)
        assert X.dtype == np.float32
        assert y.dtype == np.int32


# ---------------------------------------------------------------------------
# generate_motor_eeg — 4-class motor imagery
# ---------------------------------------------------------------------------

class TestGenerateMotorEeg:
    """Tests for data.simulate_motor_eeg.generate_motor_eeg."""

    def setup_method(self):
        from data.simulate_motor_eeg import generate_motor_eeg
        self.generate_motor_eeg = generate_motor_eeg

    def test_default_shape(self):
        X, y = self.generate_motor_eeg(n_samples=100)
        # n_timepoints = 4 * 250 = 1000
        assert X.shape == (100, 64, 1000), f"Unexpected shape: {X.shape}"
        assert y.shape == (100,)

    def test_four_classes(self):
        _, y = self.generate_motor_eeg(n_samples=400)
        classes = set(np.unique(y))
        assert classes == {0, 1, 2, 3}, f"Expected classes {{0,1,2,3}}, got {classes}"

    def test_label_values(self):
        _, y = self.generate_motor_eeg(n_samples=200)
        assert set(np.unique(y)).issubset({0, 1, 2, 3})

    def test_roughly_balanced(self):
        _, y = self.generate_motor_eeg(n_samples=400)
        for cls in range(4):
            count = int((y == cls).sum())
            assert 80 <= count <= 120, f"Class {cls} count {count} out of expected range"

    def test_custom_channels(self):
        X, _ = self.generate_motor_eeg(n_samples=10, n_channels=32)
        assert X.shape[1] == 32

    def test_dtype(self):
        X, y = self.generate_motor_eeg(n_samples=8)
        assert X.dtype == np.float32
        assert y.dtype == np.int32


# ---------------------------------------------------------------------------
# 5-class simulation (via API layer)
# ---------------------------------------------------------------------------

class TestGenerateMulticlassEeg:
    """Tests for the 5-class multiclass simulation used in api/main.py."""

    def test_five_states_returned(self):
        """The API simulation should cover all 5 brain states."""
        from api.main import _simulate_multiclass_sample, BRAIN_STATES

        assert len(BRAIN_STATES) == 5

        observed_states = set()
        for _ in range(100):
            _, state, _, _ = _simulate_multiclass_sample()
            observed_states.add(state.lower())

        # All 5 states should appear in 100 trials
        assert observed_states == set(BRAIN_STATES), (
            f"Missing states: {set(BRAIN_STATES) - observed_states}"
        )

    def test_probs_sum_to_one(self):
        from api.main import _simulate_multiclass_sample

        for _ in range(20):
            _, _, _, probs = _simulate_multiclass_sample()
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-5, f"Probs don't sum to 1: {total}"

    def test_confidence_in_range(self):
        from api.main import _simulate_multiclass_sample

        for _ in range(20):
            _, _, confidence, _ = _simulate_multiclass_sample()
            assert 0.0 <= confidence <= 1.0
