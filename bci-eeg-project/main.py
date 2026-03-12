"""
main.py
-------
CLI entry point for the BCI EEG Focus vs Relax Classifier.

Usage
-----
  python main.py --mode train     # Generate data, preprocess, train model
  python main.py --mode evaluate  # Evaluate model performance
  python main.py --mode predict   # Run real-time prediction demo
  python main.py --mode analyze   # Run EEG signal analysis tool
  python main.py --mode all       # Run all three steps in sequence
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


def _run_evaluate() -> None:
    from model.evaluate import evaluate
    evaluate()


def _run_predict() -> None:
    from inference.predict import demo
    demo()


def _run_analyze() -> None:
    from tools.eeg_analysis import summary, export, plot
    print("--- Band-Power Summary ---")
    summary()
    print("\n--- Exporting features to features.csv ---")
    export()
    print("\n--- Plotting 3 sample EEG epochs ---")
    plot(n_plot_samples=3)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bci-eeg",
        description="BCI Neural Network — EEG Focus vs Relax Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode train\n"
            "  python main.py --mode evaluate\n"
            "  python main.py --mode predict\n"
            "  python main.py --mode analyze\n"
            "  python main.py --mode all\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict", "analyze", "all"],
        required=True,
        help="Pipeline mode to run.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        print("=" * 60)
        print(" Mode: TRAIN")
        print("=" * 60)
        _run_train()

    elif args.mode == "evaluate":
        print("=" * 60)
        print(" Mode: EVALUATE")
        print("=" * 60)
        _run_evaluate()

    elif args.mode == "predict":
        print("=" * 60)
        print(" Mode: PREDICT")
        print("=" * 60)
        _run_predict()

    elif args.mode == "analyze":
        print("=" * 60)
        print(" Mode: ANALYZE")
        print("=" * 60)
        _run_analyze()

    elif args.mode == "all":
        print("=" * 60)
        print(" Mode: ALL  (train → evaluate → predict)")
        print("=" * 60)
        print("\n--- Step 1: Train ---")
        _run_train()
        print("\n--- Step 2: Evaluate ---")
        _run_evaluate()
        print("\n--- Step 3: Predict ---")
        _run_predict()


if __name__ == "__main__":
    main()
