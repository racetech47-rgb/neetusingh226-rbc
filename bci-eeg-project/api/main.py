"""
api/main.py
-----------
FastAPI application for the BCI EEG classifier.

Endpoints
---------
GET  /         → health check (redirect to /health)
GET  /health   → {"status": "ok"}
GET  /states   → list of supported brain states
POST /predict  → classify a raw EEG sample
WS   /ws       → WebSocket: streams real-time predictions every 200 ms

Usage
-----
    uvicorn api.main:app --reload
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages can be imported
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent        # bci-eeg-project/api/
_ROOT = _HERE.parent                           # bci-eeg-project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_eeg import generate_eeg_data
from preprocessing.filter import extract_features

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BCI EEG API",
    description="Real-time brain-state classification via REST and WebSocket.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Brain-state metadata
# ---------------------------------------------------------------------------
BRAIN_STATES: List[str] = ["focus", "relax", "stress", "sleep", "meditation"]

# Frequency profiles used when simulating each multiclass state
_STATE_PROFILES: Dict[str, Dict] = {
    "focus":      {"dominant": (13.0, 30.0), "secondary": (8.0, 13.0)},
    "relax":      {"dominant": (8.0,  13.0), "secondary": (4.0,  8.0)},
    "stress":     {"dominant": (20.0, 35.0), "secondary": (13.0, 20.0)},
    "sleep":      {"dominant": (0.5,   4.0), "secondary": (4.0,  8.0)},
    "meditation": {"dominant": (8.0,  10.0), "secondary": (4.0,  8.0)},
}


def _simulate_multiclass_sample(
    n_channels: int = 8,
    duration: float = 2.0,
    fs: int = 256,
) -> tuple:
    """Generate a single simulated multiclass EEG epoch.

    Returns:
        epoch   : numpy array (n_channels, n_timepoints)
        state   : predicted brain-state label (string)
        probs   : dict mapping state name → probability
    """
    # Pick a random state
    state_name = np.random.choice(BRAIN_STATES)
    profile = _STATE_PROFILES[state_name]
    n_timepoints = int(duration * fs)
    t = np.linspace(0, duration, n_timepoints, endpoint=False)

    epoch = np.zeros((n_channels, n_timepoints), dtype=np.float32)
    for ch in range(n_channels):
        lo, hi = profile["dominant"]
        freq = np.random.uniform(lo, hi)
        epoch[ch] += np.random.uniform(1.5, 3.0) * np.sin(2 * np.pi * freq * t)

        lo2, hi2 = profile["secondary"]
        freq2 = np.random.uniform(lo2, hi2)
        epoch[ch] += np.random.uniform(0.3, 0.7) * np.sin(2 * np.pi * freq2 * t)

        epoch[ch] += np.random.normal(0, 0.3, n_timepoints)

    # Build probability distribution: dominant state gets high prob, others share rest
    main_prob = np.random.uniform(0.80, 0.97)
    remaining = 1.0 - main_prob
    other_probs = np.random.dirichlet(np.ones(len(BRAIN_STATES) - 1)) * remaining
    probs: Dict[str, float] = {}
    j = 0
    for s in BRAIN_STATES:
        if s == state_name:
            probs[s] = float(main_prob)
        else:
            probs[s] = float(other_probs[j])
            j += 1

    return epoch, state_name.upper(), main_prob, probs


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for /predict."""
    eeg_data: List[List[float]]  # shape: (n_channels, n_timepoints)
    fs: int = 256


@app.get("/")
async def root():
    return {"message": "BCI EEG API — visit /docs for Swagger UI"}


@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/states")
async def get_states():
    """Return all supported brain states."""
    return {"states": [s.upper() for s in BRAIN_STATES]}


@app.post("/predict")
async def predict(req: PredictRequest):
    """Classify a raw EEG sample (array of channels × timepoints).

    Returns the predicted brain state and per-state probabilities.
    """
    epoch = np.array(req.eeg_data, dtype=np.float32)
    if epoch.ndim != 2:
        return {"error": "eeg_data must be 2-D (n_channels × n_timepoints)"}

    # Wrap in batch dim for feature extraction
    batch = epoch[np.newaxis, ...]
    features, scaler = extract_features(batch, fs=req.fs, normalize=True)

    # Use simulated prediction (no trained model required for demo)
    _, state, confidence, all_probs = _simulate_multiclass_sample(
        n_channels=epoch.shape[0], fs=req.fs
    )

    return {
        "state":      state,
        "confidence": round(confidence, 4),
        "all_probs":  {k: round(v, 4) for k, v in all_probs.items()},
        "timestamp":  int(time.time()),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Stream real-time brain-state predictions at ~5 Hz (every 200 ms).

    Sends JSON messages:
    {
        "state":      "FOCUS",
        "confidence": 0.943,
        "all_probs":  {"focus": 0.943, "relax": 0.031, ...},
        "eeg_sample": [0.12, -0.34, ...],   // mean value per channel
        "timestamp":  1234567890
    }
    """
    await websocket.accept()
    try:
        while True:
            epoch, state, confidence, all_probs = _simulate_multiclass_sample()

            # Summarise epoch as per-channel means for the wire payload
            eeg_sample = [round(float(epoch[ch].mean()), 4) for ch in range(epoch.shape[0])]

            payload = {
                "state":      state,
                "confidence": round(confidence, 4),
                "all_probs":  {k: round(v, 4) for k, v in all_probs.items()},
                "eeg_sample": eeg_sample,
                "timestamp":  int(time.time()),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.2)   # 5 Hz updates
    except WebSocketDisconnect:
        pass  # Client disconnected — exit gracefully
    except Exception:
        pass  # Any other error — close gracefully
