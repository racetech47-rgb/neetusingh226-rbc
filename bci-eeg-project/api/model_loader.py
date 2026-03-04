"""
api/model_loader.py
--------------------
Singleton model loader for the BCI FastAPI service.

Loads both the binary and multiclass Keras models together with their
StandardScaler objects exactly once at application startup.  Subsequent
requests reuse the cached objects to avoid repeated disk I/O and model
initialisation overhead.
"""

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

# Resolve project root for sibling-package imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tensorflow import keras

from model.train import MODEL_PATH, SCALER_PATH
from model.train_multiclass import MULTICLASS_MODEL_PATH, MULTICLASS_SCALER_PATH

# ------------------------------------------------------------------ #
# Module-level singletons                                             #
# ------------------------------------------------------------------ #
_binary_model: Optional[keras.Model] = None
_binary_scaler = None

_multiclass_model: Optional[keras.Model] = None
_multiclass_scaler = None

_models_loaded: bool = False


def load_models() -> None:
    """Load all models and scalers from disk.

    Called once during FastAPI startup.  Safe to call multiple times — only
    loads on the first call.

    Raises:
        FileNotFoundError: If any required artefact is missing from disk.
    """
    global _binary_model, _binary_scaler
    global _multiclass_model, _multiclass_scaler
    global _models_loaded

    if _models_loaded:
        return

    # ---- Binary model ----
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        _binary_model = keras.models.load_model(str(MODEL_PATH))
        _binary_scaler = joblib.load(str(SCALER_PATH))
        print(f"[ModelLoader] Binary model loaded from {MODEL_PATH}")
    else:
        print(
            f"[ModelLoader] WARNING: Binary model not found at {MODEL_PATH}. "
            "Run `python main.py --mode train` to generate it."
        )

    # ---- Multiclass model ----
    if MULTICLASS_MODEL_PATH.exists() and MULTICLASS_SCALER_PATH.exists():
        _multiclass_model = keras.models.load_model(str(MULTICLASS_MODEL_PATH))
        _multiclass_scaler = joblib.load(str(MULTICLASS_SCALER_PATH))
        print(f"[ModelLoader] Multiclass model loaded from {MULTICLASS_MODEL_PATH}")
    else:
        print(
            f"[ModelLoader] WARNING: Multiclass model not found at "
            f"{MULTICLASS_MODEL_PATH}. "
            "Run `python main.py --mode train-multi` to generate it."
        )

    _models_loaded = True


def get_binary_model():
    """Return the loaded binary classification model.

    Returns:
        Tuple of (model, scaler) or (None, None) if not loaded.
    """
    return _binary_model, _binary_scaler


def get_multiclass_model():
    """Return the loaded multiclass classification model.

    Returns:
        Tuple of (model, scaler) or (None, None) if not loaded.
    """
    return _multiclass_model, _multiclass_scaler


def is_loaded() -> bool:
    """Return True if at least one model has been loaded."""
    return _models_loaded and (
        _binary_model is not None or _multiclass_model is not None
    )
