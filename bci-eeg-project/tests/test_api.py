"""
tests/test_api.py
-----------------
Integration tests for the FastAPI BCI endpoints.

Uses httpx (ASGI transport) to call the app in-process — no server required.

Covers:
  - GET  /health    → 200, {"status": "ok"}
  - GET  /states    → 200, all 5 states present
  - POST /predict   → 200, valid prediction response
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Import the FastAPI app (and httpx / starlette transport)
# ---------------------------------------------------------------------------
try:
    from starlette.testclient import TestClient
    from api.main import app

    _HAS_STARLETTE = True
except ImportError:
    _HAS_STARLETTE = False


pytestmark = pytest.mark.skipif(
    not _HAS_STARLETTE, reason="starlette TestClient not available — skipping API tests"
)


@pytest.fixture
def client():
    """Return a synchronous Starlette TestClient backed by the ASGI app."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


class TestStatesEndpoint:
    EXPECTED_STATES = {"FOCUS", "RELAX", "STRESS", "SLEEP", "MEDITATION"}

    def test_returns_200(self, client):
        resp = client.get("/states")
        assert resp.status_code == 200

    def test_contains_all_five_states(self, client):
        resp = client.get("/states")
        data = resp.json()
        assert "states" in data
        assert set(data["states"]) == self.EXPECTED_STATES, (
            f"Missing states: {self.EXPECTED_STATES - set(data['states'])}"
        )

    def test_exactly_five_states(self, client):
        resp = client.get("/states")
        data = resp.json()
        assert len(data["states"]) == 5


class TestPredictEndpoint:
    def _mock_eeg(self, n_channels: int = 8, n_timepoints: int = 512) -> list:
        """Generate a mock EEG sample as a nested list."""
        data = np.random.randn(n_channels, n_timepoints).tolist()
        return data

    def test_returns_200(self, client):
        payload = {"eeg_data": self._mock_eeg(), "fs": 256}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        payload = {"eeg_data": self._mock_eeg(), "fs": 256}
        resp = client.post("/predict", json=payload)
        data = resp.json()
        for key in ("state", "confidence", "all_probs", "timestamp"):
            assert key in data, f"Missing key: {key}"

    def test_confidence_in_valid_range(self, client):
        payload = {"eeg_data": self._mock_eeg(), "fs": 256}
        resp = client.post("/predict", json=payload)
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_state_is_valid(self, client):
        valid_states = {"FOCUS", "RELAX", "STRESS", "SLEEP", "MEDITATION"}
        payload = {"eeg_data": self._mock_eeg(), "fs": 256}
        resp = client.post("/predict", json=payload)
        state = resp.json()["state"]
        assert state in valid_states, f"Invalid state: {state}"

    def test_all_probs_sum_to_one(self, client):
        payload = {"eeg_data": self._mock_eeg(), "fs": 256}
        resp = client.post("/predict", json=payload)
        probs = resp.json()["all_probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities don't sum to 1: {total}"
