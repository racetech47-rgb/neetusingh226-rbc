"""
main.py
-------
CLI entry point for the BCI EEG Classifier.

Usage
-----
  python main.py --mode train           # original binary classifier
  python main.py --mode train-multi     # new 5-class classifier
  python main.py --mode evaluate        # evaluate binary model
  python main.py --mode evaluate-multi  # evaluate multiclass model
  python main.py --mode predict         # binary prediction demo
  python main.py --mode predict-multi   # multiclass prediction demo
  python main.py --mode dashboard       # launch live real-time dashboard
  python main.py --mode export-onnx     # export multiclass model to ONNX
  python main.py --mode api             # start FastAPI REST server
  python main.py --mode all             # run all steps in sequence
"""

import argparse
import sys
from pathlib import Path

# Make sure project root is on the path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _run_train() -> None:
    from model.train import train
    train()


def _run_train_multi() -> None:
    from model.train_multiclass import train_multiclass
    train_multiclass()


def _run_evaluate() -> None:
    from model.evaluate import evaluate
    evaluate()


def _run_evaluate_multi() -> None:
    """Evaluate the multiclass model on freshly generated test data."""
    import numpy as np
    import joblib
    from sklearn.metrics import classification_report
    from tensorflow import keras
    from model.train_multiclass import (
        MULTICLASS_MODEL_PATH,
        MULTICLASS_SCALER_PATH,
        CLASS_NAMES,
    )
    from data.simulate_eeg import generate_multiclass_eeg
    from preprocessing.filter import extract_features

    if not MULTICLASS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Multiclass model not found at {MULTICLASS_MODEL_PATH}. "
            "Run `python main.py --mode train-multi` first."
        )

    model = keras.models.load_model(str(MULTICLASS_MODEL_PATH))
    scaler = joblib.load(str(MULTICLASS_SCALER_PATH))

    print("\nGenerating evaluation samples …")
    signals, labels = generate_multiclass_eeg(n_samples=500)
    features, _ = extract_features(signals, normalize=False)
    features = scaler.transform(features)

    y_pred = np.argmax(model.predict(features, verbose=0), axis=1)

    print("\n=== Multiclass Classification Report ===")
    print(classification_report(labels, y_pred, target_names=CLASS_NAMES, digits=4))


def _run_predict() -> None:
    from inference.predict import demo
    demo()


def _run_predict_multi() -> None:
    from inference.predict_multiclass import demo
    demo()


def _run_dashboard() -> None:
    from dashboard.live_plot import start_dashboard
    start_dashboard(use_real_hardware=False)


def _run_export_onnx() -> None:
    from export.export_onnx import export_to_onnx
    export_to_onnx()


def _run_api() -> None:
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to start the API server. "
            "Install with: pip install uvicorn fastapi"
        )
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=str(_ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bci-eeg",
        description="BCI Neural Network — EEG Brain State Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode train\n"
            "  python main.py --mode train-multi\n"
            "  python main.py --mode evaluate\n"
            "  python main.py --mode evaluate-multi\n"
            "  python main.py --mode predict\n"
            "  python main.py --mode predict-multi\n"
            "  python main.py --mode dashboard\n"
            "  python main.py --mode export-onnx\n"
            "  python main.py --mode api\n"
            "  python main.py --mode all\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "train-multi",
            "evaluate",
            "evaluate-multi",
            "predict",
            "predict-multi",
            "dashboard",
            "export-onnx",
            "api",
            "all",
        ],
        required=True,
        help="Pipeline mode to run.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Mode dispatch                                                         #
    # ------------------------------------------------------------------ #
    if args.mode == "train":
        print("=" * 60)
        print(" Mode: TRAIN (binary)")
        print("=" * 60)
        _run_train()

    elif args.mode == "train-multi":
        print("=" * 60)
        print(" Mode: TRAIN-MULTI (5-class)")
        print("=" * 60)
        _run_train_multi()

    elif args.mode == "evaluate":
        print("=" * 60)
        print(" Mode: EVALUATE (binary)")
        print("=" * 60)
        _run_evaluate()

    elif args.mode == "evaluate-multi":
        print("=" * 60)
        print(" Mode: EVALUATE-MULTI (5-class)")
        print("=" * 60)
        _run_evaluate_multi()

    elif args.mode == "predict":
        print("=" * 60)
        print(" Mode: PREDICT (binary)")
        print("=" * 60)
        _run_predict()

    elif args.mode == "predict-multi":
        print("=" * 60)
        print(" Mode: PREDICT-MULTI (5-class)")
        print("=" * 60)
        _run_predict_multi()

    elif args.mode == "dashboard":
        print("=" * 60)
        print(" Mode: DASHBOARD (live real-time)")
        print("=" * 60)
        _run_dashboard()

    elif args.mode == "export-onnx":
        print("=" * 60)
        print(" Mode: EXPORT-ONNX")
        print("=" * 60)
        _run_export_onnx()

    elif args.mode == "api":
        print("=" * 60)
        print(" Mode: API (FastAPI server on http://0.0.0.0:8000)")
        print("=" * 60)
        _run_api()

    elif args.mode == "all":
        print("=" * 60)
        print(" Mode: ALL  (train → train-multi → evaluate → predict → predict-multi)")
        print("=" * 60)
        print("\n--- Step 1: Train (binary) ---")
        _run_train()
        print("\n--- Step 2: Train multiclass ---")
        _run_train_multi()
        print("\n--- Step 3: Evaluate (binary) ---")
        _run_evaluate()
        print("\n--- Step 4: Evaluate multiclass ---")
        _run_evaluate_multi()
        print("\n--- Step 5: Predict (binary) ---")
        _run_predict()
        print("\n--- Step 6: Predict multiclass ---")
        _run_predict_multi()


if __name__ == "__main__":
    main()
