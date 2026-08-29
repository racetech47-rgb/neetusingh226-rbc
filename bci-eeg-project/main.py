"""
main.py
-------
CLI entry point for the BCI EEG Classifier (Phase 3).

Usage
-----
  python main.py --mode train           # Generate data, preprocess, train model
  python main.py --mode evaluate        # Evaluate model performance
  python main.py --mode predict         # Run real-time prediction demo
  python main.py --mode webapp          # Start React webapp + API together
  python main.py --mode docker          # Print docker-compose up instructions
  python main.py --mode download-data   # Download PhysioNet EEG dataset
  python main.py --mode finetune        # Fine-tune on PhysioNet data
  python main.py --mode train-motor     # Train motor imagery classifier
  python main.py --mode predict-motor   # Run motor imagery prediction demo
  python main.py --mode bci-pong        # Launch BCI Pong game
  python main.py --mode all             # Run train → evaluate → predict
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Make sure project root is on the path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _run_train() -> None:
    from model.train import train
    train()


def _run_evaluate() -> None:
    from model.evaluate import evaluate
    evaluate()


def _run_predict() -> None:
    from inference.predict import demo
    demo()


def _run_webapp() -> None:
    """Start the FastAPI backend and print instructions for the React webapp."""
    print("\n🌐 Starting BCI API backend …")
    print("   Run the React webapp separately with:")
    print("   cd ../webapp && npm install && npm start\n")
    print("   API docs: http://localhost:8000/docs")
    print("   WebSocket: ws://localhost:8000/ws\n")
    try:
        subprocess.run(
            ["uvicorn", "api.main:app", "--reload"],
            check=True,
        )
    except FileNotFoundError:
        print("uvicorn not found. Install with: pip install uvicorn[standard]")
    except KeyboardInterrupt:
        print("\nAPI server stopped.")


def _run_docker() -> None:
    """Print docker-compose instructions."""
    print("\n🐳 Docker Deployment")
    print("=" * 50)
    print("Run the full stack with:\n")
    print("  cd bci-eeg-project")
    print("  docker-compose up --build\n")
    print("Services:")
    print("  BCI API      → http://localhost:8000")
    print("  Web Dashboard → http://localhost:3000\n")
    print("To stop:  docker-compose down")


def _run_download_data() -> None:
    """Download the PhysioNet EEG Motor Imagery dataset."""
    from datasets.physionet_loader import download_physionet
    download_physionet(subject_ids=[1, 2, 3])


def _run_finetune() -> None:
    """Fine-tune the pre-trained model on PhysioNet data."""
    from model.finetune import finetune_on_physionet
    finetune_on_physionet(subjects=[1, 2, 3])


def _run_train_motor() -> None:
    """Train the motor imagery classifier."""
    from model.train_motor import train_motor
    train_motor()


def _run_predict_motor() -> None:
    """Run the motor imagery prediction demo."""
    from inference.predict_motor import demo_motor
    demo_motor()


def _run_bci_pong() -> None:
    """Launch the BCI Pong game."""
    from games.bci_pong import start_bci_pong
    start_bci_pong()


def main() -> None:
    _MODES = [
        "train",
        "evaluate",
        "predict",
        "webapp",
        "docker",
        "download-data",
        "finetune",
        "train-motor",
        "predict-motor",
        "bci-pong",
        "all",
    ]

    parser = argparse.ArgumentParser(
        prog="bci-eeg",
        description="BCI Neural Network — EEG Brain-State Classifier (Phase 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode train\n"
            "  python main.py --mode evaluate\n"
            "  python main.py --mode predict\n"
            "  python main.py --mode webapp\n"
            "  python main.py --mode docker\n"
            "  python main.py --mode download-data\n"
            "  python main.py --mode finetune\n"
            "  python main.py --mode train-motor\n"
            "  python main.py --mode predict-motor\n"
            "  python main.py --mode bci-pong\n"
            "  python main.py --mode all\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=_MODES,
        required=True,
        help="Pipeline mode to run.",
    )
    args = parser.parse_args()

    _HANDLERS = {
        "train":         ("TRAIN",          _run_train),
        "evaluate":      ("EVALUATE",       _run_evaluate),
        "predict":       ("PREDICT",        _run_predict),
        "webapp":        ("WEBAPP",         _run_webapp),
        "docker":        ("DOCKER",         _run_docker),
        "download-data": ("DOWNLOAD DATA",  _run_download_data),
        "finetune":      ("FINETUNE",       _run_finetune),
        "train-motor":   ("TRAIN MOTOR",    _run_train_motor),
        "predict-motor": ("PREDICT MOTOR",  _run_predict_motor),
        "bci-pong":      ("BCI PONG",       _run_bci_pong),
    }

    if args.mode == "all":
        print("=" * 60)
        print(" Mode: ALL  (train → evaluate → predict)")
        print("=" * 60)
        print("\n--- Step 1: Train ---")
        _run_train()
        print("\n--- Step 2: Evaluate ---")
        _run_evaluate()
        print("\n--- Step 3: Predict ---")
        _run_predict()
    else:
        label, handler = _HANDLERS[args.mode]
        print("=" * 60)
        print(f" Mode: {label}")
        print("=" * 60)
        handler()


if __name__ == "__main__":
    main()
