"""
api/schemas.py
---------------
Pydantic request/response models for the BCI FastAPI REST API.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class EEGInput(BaseModel):
    """Input schema for /predict and /predict/multiclass endpoints.

    Attributes:
        eeg_data: List of EEG epochs; each epoch is a list of channel
                  time-series lists.  Shape: (n_epochs, n_channels, n_timepoints).
                  Typically one epoch is sent per request.
    """

    eeg_data: List[List[List[float]]] = Field(
        ...,
        description=(
            "EEG data as a nested list of shape "
            "(n_epochs, n_channels, n_timepoints)."
        ),
        example=[[[0.1, -0.2, 0.3]] * 512],
    )


class PredictionResponse(BaseModel):
    """Response schema for the binary /predict endpoint.

    Attributes:
        state:      Predicted brain state (FOCUS or RELAX).
        confidence: Model confidence as a value in [0, 1].
    """

    state: str = Field(..., example="FOCUS")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.943)


class MulticlassPredictionResponse(BaseModel):
    """Response schema for the /predict/multiclass endpoint.

    Attributes:
        state:      Predicted brain state name (upper-case).
        confidence: Confidence for the winning class in [0, 1].
        all_probs:  Softmax probability for every class.
    """

    state: str = Field(..., example="MEDITATION")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.912)
    all_probs: Dict[str, float] = Field(
        ...,
        example={
            "focus": 0.012,
            "relax": 0.031,
            "stress": 0.008,
            "sleep": 0.037,
            "meditation": 0.912,
        },
    )
