"""
datasets/physionet_loader.py
-----------------------------
PhysioNet EEG Motor Movement/Imagery Dataset downloader and feature extractor.

Uses MNE-Python to automatically download and parse the dataset, then extracts
features compatible with the existing BCI preprocessing pipeline.

Motor imagery events used:
  T0 → rest
  T1 → left-hand movement / imagery
  T2 → right-hand movement / imagery

Usage
-----
    from datasets.physionet_loader import (
        download_physionet,
        load_epochs,
        extract_physionet_features,
    )
    download_physionet(subject_ids=[1, 2, 3])
    epochs = load_epochs(subject_id=1)
    X, y = extract_physionet_features(epochs)
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project-root import guard
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent        # bci-eeg-project/datasets/
_ROOT = _HERE.parent                           # bci-eeg-project/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.filter import extract_features as _extract_features


# PhysioNet EEG tasks that contain motor imagery (runs 6,10,14 = imagery)
_IMAGERY_RUNS: List[int] = [6, 10, 14]

# Event mapping for motor imagery tasks
_EVENT_ID: dict = {"rest": 1, "left_hand": 2, "right_hand": 3}


def download_physionet(
    subject_ids: List[int] = None,
    data_dir: str = "datasets/physionet/",
) -> None:
    """Download PhysioNet EEG Motor Movement/Imagery data using MNE.

    Args:
        subject_ids: List of subject IDs to download (1-109).
                     Defaults to [1, 2, 3].
        data_dir:    Directory where the data will be stored.
    """
    if subject_ids is None:
        subject_ids = [1, 2, 3]

    try:
        import mne  # type: ignore
        from mne.datasets import eegbci  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MNE-Python is required. Install with: pip install mne"
        ) from exc

    dest = Path(data_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for sid in subject_ids:
        print(f"[PhysioNet] Downloading subject {sid} …")
        # fetch_data returns a list of file paths
        eegbci.load_data(sid, runs=_IMAGERY_RUNS, path=str(dest), verbose=False)
        print(f"[PhysioNet] Subject {sid} downloaded.")


def load_epochs(
    subject_id: int,
    data_dir: str = "datasets/physionet/",
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> "mne.Epochs":  # type: ignore  # noqa: F821
    """Load MNE Epochs for a single subject.

    Args:
        subject_id: Subject number (1-109).
        data_dir:   Root directory where PhysioNet data is stored.
        tmin:       Epoch start time relative to event onset (seconds).
        tmax:       Epoch end time relative to event onset (seconds).

    Returns:
        MNE Epochs object.
    """
    try:
        import mne  # type: ignore
        from mne.datasets import eegbci  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MNE-Python is required. Install with: pip install mne"
        ) from exc

    files = eegbci.load_data(
        subject_id, runs=_IMAGERY_RUNS, path=data_dir, verbose=False
    )

    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = mne.concatenate_raws(raws)

    # Standardise channel names to match 10-20 system
    eegbci.standardize(raw)

    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Map T0/T1/T2 to rest/left_hand/right_hand
    event_id_mapped = {
        k: v for k, v in event_id.items() if k in ("T0", "T1", "T2")
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_mapped,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        baseline=None,
        preload=True,
        verbose=False,
    )

    print(f"[PhysioNet] Subject {subject_id}: {len(epochs)} epochs, "
          f"{len(epochs.ch_names)} channels, {epochs.info['sfreq']:.0f} Hz sample rate")

    return epochs


def extract_physionet_features(
    epochs: "mne.Epochs",  # type: ignore  # noqa: F821
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frequency-band features compatible with the BCI pipeline.

    Converts MNE Epochs data (n_epochs, n_channels, n_timepoints) into a
    feature matrix using the existing bandpass + FFT band-power pipeline.

    Args:
        epochs: MNE Epochs object.

    Returns:
        X: Feature matrix, shape (n_epochs, n_channels * n_bands).
        y: Integer labels (0=rest, 1=left_hand, 2=right_hand).
    """
    data = epochs.get_data(units="uV")        # (n_epochs, n_channels, n_timepoints)
    fs   = int(epochs.info["sfreq"])

    # Use only the first 8 channels to match the existing pipeline
    n_channels_used = min(8, data.shape[1])
    data = data[:, :n_channels_used, :].astype(np.float32)

    features, _ = _extract_features(data, fs=fs, normalize=True)

    # Build label array (0=T0/rest, 1=T1/left, 2=T2/right)
    label_map = {"T0": 0, "T1": 1, "T2": 2}
    raw_labels = epochs.events[:, 2]
    event_id_inv = {v: k for k, v in epochs.event_id.items()}
    labels = np.array(
        [label_map.get(event_id_inv.get(ev, "T0"), 0) for ev in raw_labels],
        dtype=np.int32,
    )

    print(f"[PhysioNet] Features: {features.shape}, Labels: {labels.shape}")
    return features, labels
