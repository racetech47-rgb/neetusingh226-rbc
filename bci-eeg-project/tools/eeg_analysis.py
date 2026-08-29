"""
tools/eeg_analysis.py
---------------------
EEG Signal Analysis Tool for the BCI Focus vs Relax Classifier.

Provides command-line utilities for:
  - Summarising band-power statistics across a generated EEG dataset.
  - Exporting feature matrices to CSV for offline analysis.
  - Plotting sample EEG waveforms and their frequency spectra.

Usage (standalone)
------------------
  python tools/eeg_analysis.py --action summary
  python tools/eeg_analysis.py --action export --output features.csv
  python tools/eeg_analysis.py --action plot   --samples 3
  python tools/eeg_analysis.py --action all
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in all envs
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on the path when run as a script
_HERE = Path(__file__).resolve().parent          # bci-eeg-project/tools/
_ROOT = _HERE.parent                             # bci-eeg-project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.simulate_eeg import generate_eeg_data
from preprocessing.filter import BANDS, extract_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_names() -> list:
    """Return ordered list of band names from the BANDS dict."""
    return list(BANDS.keys())


def _feature_column_names(n_channels: int) -> list:
    """Build column headers: ch0_theta, ch0_alpha, …, ch7_beta."""
    cols = []
    for ch in range(n_channels):
        for band in _band_names():
            cols.append(f"ch{ch}_{band}")
    return cols


# ---------------------------------------------------------------------------
# Action: summary
# ---------------------------------------------------------------------------

def summary(n_samples: int = 200) -> None:
    """Print band-power statistics (mean ± std) for each class.

    Args:
        n_samples: Number of EEG epochs to generate (default: 200).
    """
    print(f"\nGenerating {n_samples} EEG epochs …")
    signals, labels = generate_eeg_data(n_samples=n_samples)
    features, _ = extract_features(signals, normalize=False)

    n_channels = signals.shape[1]
    col_names = _feature_column_names(n_channels)

    class_names = {0: "Relax", 1: "Focus"}

    print("\n" + "=" * 60)
    print(" EEG Band-Power Feature Summary")
    print("=" * 60)

    for cls_id, cls_name in class_names.items():
        mask = labels == cls_id
        cls_features = features[mask]
        print(f"\n  Class: {cls_name}  ({mask.sum()} samples)")
        print(f"  {'Feature':<20}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
        print("  " + "-" * 64)
        for i, col in enumerate(col_names):
            col_data = cls_features[:, i]
            print(
                f"  {col:<20}  {col_data.mean():>10.4f}  {col_data.std():>10.4f}"
                f"  {col_data.min():>10.4f}  {col_data.max():>10.4f}"
            )

    print()


# ---------------------------------------------------------------------------
# Action: export
# ---------------------------------------------------------------------------

def export(output_path: str = "features.csv", n_samples: int = 200) -> None:
    """Export feature matrix and labels to a CSV file.

    Args:
        output_path: Destination CSV file path (default: features.csv).
        n_samples:   Number of EEG epochs to generate (default: 200).
    """
    print(f"\nGenerating {n_samples} EEG epochs …")
    signals, labels = generate_eeg_data(n_samples=n_samples)
    features, _ = extract_features(signals, normalize=False)

    n_channels = signals.shape[1]
    col_names = _feature_column_names(n_channels)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_names + ["label", "class_name"])
        class_names = {0: "Relax", 1: "Focus"}
        for feat_row, label in zip(features, labels):
            writer.writerow(list(feat_row) + [int(label), class_names[int(label)]])

    print(f"✅ Feature matrix exported → {out.resolve()}")
    print(f"   Rows: {len(labels)}  |  Columns: {len(col_names) + 2}")


# ---------------------------------------------------------------------------
# Action: plot
# ---------------------------------------------------------------------------

def plot(n_plot_samples: int = 3, output_dir: str = "plots") -> None:
    """Plot sample EEG waveforms and their FFT power spectra.

    One figure is saved per requested sample, showing:
      - Upper panel: raw 8-channel EEG waveform.
      - Lower panel: FFT power spectrum for each channel with band
        boundaries marked.

    Args:
        n_plot_samples: Number of sample epochs to plot (default: 3).
        output_dir:     Directory where PNG files are saved (default: plots/).
    """
    total = max(n_plot_samples * 2, 10)     # generate more than needed
    print(f"\nGenerating EEG data for plotting …")
    signals, labels = generate_eeg_data(n_samples=total)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = {0: "Relax", 1: "Focus"}
    fs = 256
    n_channels = signals.shape[1]
    n_timepoints = signals.shape[2]
    t = np.linspace(0, n_timepoints / fs, n_timepoints, endpoint=False)
    freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / fs)

    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))

    saved = []
    for sample_idx in range(min(n_plot_samples, total)):
        epoch = signals[sample_idx]   # (n_channels, n_timepoints)
        cls_name = class_names[int(labels[sample_idx])]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        fig.suptitle(
            f"Sample {sample_idx + 1} — Class: {cls_name}", fontsize=14, fontweight="bold"
        )

        # --- Raw waveform ---
        ax_wave = axes[0]
        for ch in range(n_channels):
            offset = ch * 6.0   # vertical separation between channels
            ax_wave.plot(t, epoch[ch] + offset, color=colors[ch], linewidth=0.8,
                         label=f"Ch {ch}")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude + offset (µV)")
        ax_wave.set_title("Raw EEG Waveform")
        ax_wave.legend(loc="upper right", fontsize=7, ncol=4)
        ax_wave.grid(alpha=0.3)

        # --- FFT power spectrum ---
        ax_fft = axes[1]
        for ch in range(n_channels):
            fft_vals = np.fft.rfft(epoch[ch])
            power = np.abs(fft_vals) ** 2
            ax_fft.plot(freqs, power, color=colors[ch], linewidth=0.7, alpha=0.8)

        # Mark band boundaries
        band_colors = {"theta": "green", "alpha": "orange", "beta": "red"}
        for band_name, (low, high) in BANDS.items():
            ax_fft.axvspan(low, high, alpha=0.08, color=band_colors[band_name],
                           label=f"{band_name} ({low}–{high} Hz)")

        ax_fft.set_xlim(0, 50)
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Power")
        ax_fft.set_title("FFT Power Spectrum")
        ax_fft.legend(loc="upper right", fontsize=8)
        ax_fft.grid(alpha=0.3)

        fig.tight_layout()
        out_path = out_dir / f"sample_{sample_idx + 1}_{cls_name.lower()}.png"
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        saved.append(out_path)
        print(f"  Plot saved → {out_path.resolve()}")

    print(f"\n✅ {len(saved)} plot(s) saved to {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eeg-analysis",
        description="EEG Signal Analysis Tool — BCI Focus vs Relax Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/eeg_analysis.py --action summary\n"
            "  python tools/eeg_analysis.py --action export --output features.csv\n"
            "  python tools/eeg_analysis.py --action plot   --samples 3\n"
            "  python tools/eeg_analysis.py --action all\n"
        ),
    )
    parser.add_argument(
        "--action",
        choices=["summary", "export", "plot", "all"],
        required=True,
        help="Analysis action to perform.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of EEG epochs to generate (default: 200).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.csv",
        help="Output CSV path for the export action (default: features.csv).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots",
        dest="plot_dir",
        help="Output directory for plot PNG files (default: plots/).",
    )
    args = parser.parse_args()

    if args.action == "summary":
        summary(n_samples=args.samples)

    elif args.action == "export":
        export(output_path=args.output, n_samples=args.samples)

    elif args.action == "plot":
        plot(n_plot_samples=args.samples, output_dir=args.plot_dir)

    elif args.action == "all":
        print("=== Summary ===")
        summary(n_samples=args.samples)
        print("\n=== Export ===")
        export(output_path=args.output, n_samples=args.samples)
        print("\n=== Plot ===")
        plot(n_plot_samples=3, output_dir=args.plot_dir)


if __name__ == "__main__":
    main()
