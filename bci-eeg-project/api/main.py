"""
api/main.py
------------
FastAPI REST API for the BCI EEG Brain State Classifier.

Endpoints
---------
  GET  /health              → { status: "ok", model: "loaded" }
  POST /predict             → binary classifier (FOCUS / RELAX)
  POST /predict/multiclass  → 5-class classifier with per-class probabilities
  GET  /states              → list of all brain states with descriptions
  GET  /docs                → Swagger UI (built into FastAPI)
  GET  /redoc               → ReDoc UI

Start the server:
  uvicorn api.main:app --reload --app-dir /path/to/bci-eeg-project

Or via main.py:
  python main.py --mode api
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List

import numpy as np

# Resolve project root for sibling-package imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import EEGInput, PredictionResponse, MulticlassPredictionResponse
import api.model_loader as model_loader
from preprocessing.filter import extract_features
from data.simulate_eeg import BRAIN_STATES

# ------------------------------------------------------------------ #
# Lifespan context manager (startup / shutdown)                        #
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models on startup and clean up on shutdown."""
    model_loader.load_models()
    yield
    # No teardown needed — OS will release resources on process exit


# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="BCI EEG Brain State Classifier API",
    description=(
        "REST API for classifying brain states from EEG data using "
        "trained neural network models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins so the API can be called from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Human-readable state metadata
_STATE_INFO = {
    "focus":      "Active thinking, alertness — dominant beta (13–30 Hz)",
    "relax":      "Calm, idle state — dominant alpha (8–13 Hz)",
    "stress":     "Heightened arousal — high beta + gamma (25–100 Hz)",
    "sleep":      "Drowsiness / deep sleep — delta + theta (0.5–8 Hz)",
    "meditation": "Focused calm — theta + alpha (4–13 Hz)",
}

# Label order for the 5-class model (must match training)
_CLASS_NAMES = [BRAIN_STATES[i] for i in sorted(BRAIN_STATES)]



# ------------------------------------------------------------------ #
# Health check                                                         #
# ------------------------------------------------------------------ #
@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """Return the API status and whether the model is loaded.

    Returns:
        JSON with ``status`` and ``model`` fields.
    """
    status = "loaded" if model_loader.is_loaded() else "not_loaded"
    return {"status": "ok", "model": status}


# ------------------------------------------------------------------ #
# Binary prediction                                                    #
# ------------------------------------------------------------------ #
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Binary EEG classification (FOCUS / RELAX)",
)
async def predict(body: EEGInput) -> PredictionResponse:
    """Classify the first EEG epoch in *eeg_data* as FOCUS or RELAX.

    Args:
        body: JSON with an ``eeg_data`` field of shape
              ``(n_epochs, n_channels, n_timepoints)``.

    Returns:
        Predicted state and confidence.
    """
    bin_model, bin_scaler = model_loader.get_binary_model()
    if bin_model is None or bin_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Binary model not loaded. Run `python main.py --mode train` first.",
        )

    epoch = np.array(body.eeg_data[0], dtype=np.float32)  # (n_channels, n_timepoints)
    features, _ = extract_features(epoch[np.newaxis, ...], normalize=False)
    features_scaled = bin_scaler.transform(features)

    prob: float = float(bin_model.predict(features_scaled, verbose=0).flatten()[0])

    if prob >= 0.5:
        state, confidence = "FOCUS", prob
    else:
        state, confidence = "RELAX", 1.0 - prob

    return PredictionResponse(state=state, confidence=round(confidence, 6))


# ------------------------------------------------------------------ #
# Multiclass prediction                                                #
# ------------------------------------------------------------------ #
@app.post(
    "/predict/multiclass",
    response_model=MulticlassPredictionResponse,
    summary="5-class EEG classification (focus/relax/stress/sleep/meditation)",
)
async def predict_multiclass(body: EEGInput) -> MulticlassPredictionResponse:
    """Classify the first EEG epoch in *eeg_data* into one of 5 brain states.

    Args:
        body: JSON with an ``eeg_data`` field of shape
              ``(n_epochs, n_channels, n_timepoints)``.

    Returns:
        Predicted state, confidence, and full probability distribution.
    """
    mc_model, mc_scaler = model_loader.get_multiclass_model()
    if mc_model is None or mc_scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Multiclass model not loaded. "
                "Run `python main.py --mode train-multi` first."
            ),
        )

    epoch = np.array(body.eeg_data[0], dtype=np.float32)
    features, _ = extract_features(epoch[np.newaxis, ...], normalize=False)
    features_scaled = mc_scaler.transform(features)

    probs: np.ndarray = mc_model.predict(features_scaled, verbose=0)[0]

    best_idx = int(np.argmax(probs))
    state = _CLASS_NAMES[best_idx].upper()
    confidence = float(probs[best_idx])
    all_probs = {name: round(float(p), 6) for name, p in zip(_CLASS_NAMES, probs)}

    return MulticlassPredictionResponse(
        state=state,
        confidence=round(confidence, 6),
        all_probs=all_probs,
    )


# ------------------------------------------------------------------ #
# Brain state descriptions                                             #
# ------------------------------------------------------------------ #
@app.get("/states", summary="List all brain states with descriptions")
async def get_states() -> List[Dict[str, str]]:
    """Return a list of all recognised brain states with descriptions.

    Returns:
        List of dicts with ``id``, ``name``, and ``description`` fields.
    """
    return [
        {
            "id": str(state_id),
            "name": BRAIN_STATES[state_id],
            "description": _STATE_INFO.get(BRAIN_STATES[state_id], ""),
        }
        for state_id in sorted(BRAIN_STATES)
    ]


# ------------------------------------------------------------------ #
# Dev runner                                                           #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
