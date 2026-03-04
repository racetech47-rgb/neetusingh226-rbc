"""
export/export_onnx.py
----------------------
Export the trained multiclass BCI Keras model to ONNX format.

Requires:
    pip install tf2onnx onnx

Usage:
    python export/export_onnx.py
    # or via main.py:
    python main.py --mode export-onnx

Output:
    export/bci_model.onnx
"""

import sys
from pathlib import Path

import numpy as np

# Resolve project root for sibling-package imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ONNX_OUTPUT_PATH = _HERE / "bci_model.onnx"


def export_to_onnx() -> None:
    """Load the trained Keras multiclass model and export it to ONNX.

    Steps:
      1. Load bci_multiclass_model.h5 from model/saved_model/
      2. Convert to ONNX using tf2onnx
      3. Save to export/bci_model.onnx
      4. Print model input / output shapes
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        import tf2onnx  # type: ignore
        import onnx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tf2onnx and onnx are required for ONNX export. "
            "Install with: pip install tf2onnx onnx"
        ) from exc

    from model.train_multiclass import MULTICLASS_MODEL_PATH

    if not MULTICLASS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Multiclass model not found at {MULTICLASS_MODEL_PATH}. "
            "Run `python main.py --mode train-multi` first."
        )

    print(f"[ONNX] Loading Keras model from {MULTICLASS_MODEL_PATH} …")
    model = keras.models.load_model(str(MULTICLASS_MODEL_PATH))

    # Print Keras input/output info
    input_shape = model.input_shape
    output_shape = model.output_shape
    print(f"[ONNX] Keras input  shape : {input_shape}")
    print(f"[ONNX] Keras output shape : {output_shape}")

    # Convert using tf2onnx
    print("[ONNX] Converting to ONNX format …")
    input_signature = [
        tf.TensorSpec(
            shape=model.input_shape,
            dtype=tf.float32,
            name="eeg_features",
        )
    ]

    _HERE.mkdir(parents=True, exist_ok=True)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=str(ONNX_OUTPUT_PATH),
    )

    # Verify the saved file
    loaded = onnx.load(str(ONNX_OUTPUT_PATH))
    onnx.checker.check_model(loaded)

    # Print ONNX input/output info
    for inp in loaded.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"[ONNX] ONNX input  : name='{inp.name}'  shape={shape}")
    for out in loaded.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"[ONNX] ONNX output : name='{out.name}'  shape={shape}")

    print(f"\n✅ ONNX model saved → {ONNX_OUTPUT_PATH}")


if __name__ == "__main__":
    export_to_onnx()
