# BCI Neural Network — EEG Brain State Classifier

A Brain-Computer Interface (BCI) project that classifies **five brain states**
(focus, relax, stress, sleep, meditation) using simulated EEG
(Electroencephalography) data and deep neural networks built with
TensorFlow/Keras.  The project also supports real EEG hardware (Muse / OpenBCI),
a live real-time dashboard, ONNX export, a Node.js CLI, and a FastAPI REST API.

---

## What is BCI / EEG?

**EEG** records electrical activity produced by neurons in the brain using
electrodes placed on the scalp.  Different mental states are associated with
distinct frequency bands:

| Band   | Frequency    | Mental state                              |
|--------|--------------|-------------------------------------------|
| Delta  | 0.5–4 Hz     | Deep sleep                                |
| Theta  | 4–8 Hz       | Drowsiness, light sleep, meditation       |
| Alpha  | 8–13 Hz      | Relaxed, eyes closed, idle                |
| Beta   | 13–30 Hz     | Active thinking, focus, alertness         |
| Gamma  | 30–100 Hz    | High-level cognition, stress              |

A **Brain-Computer Interface** uses these signals to enable direct
communication between the brain and external devices — without any muscle
movement.

---

## Feature Overview

| Feature                  | Technology                            |
|--------------------------|---------------------------------------|
| Binary classifier        | Dense NN, TensorFlow/Keras            |
| 5-class classifier       | Dense NN, TensorFlow/Keras            |
| Muse headset support     | pylsl (Lab Streaming Layer)           |
| OpenBCI support          | brainflow                             |
| Live dashboard           | matplotlib FuncAnimation              |
| ONNX export              | tf2onnx                               |
| Node.js CLI              | onnxruntime-node, commander, chalk    |
| REST API                 | FastAPI, uvicorn, Pydantic            |

---

## Project Structure

```
bci-eeg-project/
├── README.md
├── requirements.txt
├── main.py                            ← CLI entry point
├── data/
│   └── simulate_eeg.py               ← EEG generator (binary + 5-class)
├── preprocessing/
│   ├── __init__.py
│   └── filter.py                     ← Bandpass filter + FFT + feature extraction
├── model/
│   ├── __init__.py
│   ├── train.py                      ← Binary classifier training
│   ├── train_multiclass.py           ← 5-class classifier training
│   ├── evaluate.py                   ← Binary evaluation + plots
│   └── saved_model/                  ← Persisted artefacts (auto-created)
│       ├── bci_model.h5
│       ├── bci_multiclass_model.h5
│       ├── scaler.pkl
│       └── scaler_multiclass.pkl
├── inference/
│   ├── __init__.py
│   ├── predict.py                    ← Binary prediction
│   └── predict_multiclass.py         ← 5-class prediction
├── hardware/
│   ├── __init__.py
│   ├── muse_reader.py                ← Muse headset via pylsl
│   └── openbci_reader.py             ← OpenBCI via brainflow
├── dashboard/
│   ├── __init__.py
│   └── live_plot.py                  ← Real-time matplotlib dashboard
├── export/
│   └── export_onnx.py                ← ONNX model export (tf2onnx)
├── nodejs_cli/
│   ├── package.json
│   ├── README.md
│   ├── bin/
│   │   └── bci.js                    ← CLI entry point
│   └── src/
│       └── inference.js              ← ONNX inference module
└── api/
    ├── main.py                       ← FastAPI app
    ├── schemas.py                    ← Pydantic models
    ├── model_loader.py               ← Singleton model loader
    ├── requirements.txt
    └── README.md
```

---

## Architecture

```
EEG Signal (8 ch × 512 pts)
         │
         ▼
┌────────────────────┐
│  Bandpass Filter   │   theta / alpha / beta / gamma bands per channel
│  (Butterworth IIR) │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  FFT Band-Power    │   8 channels × 3 bands = 24 features
│  Feature Extraction│
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  StandardScaler    │   zero-mean, unit-variance normalisation
└────────┬───────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│  Binary Classifier      │   │  5-Class Classifier          │
│  Dense(128) → Dense(64) │   │  Dense(256) → Dense(128)    │
│  Dense(1, sigmoid)      │   │  → Dense(64)                │
│  "FOCUS" / "RELAX"      │   │  Dense(5, softmax)          │
└─────────────────────────┘   │  focus/relax/stress/         │
                               │  sleep/meditation            │
                               └─────────────────────────────┘
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/racetech47-rgb/neetusingh226-rbc.git
cd neetusingh226-rbc/bci-eeg-project

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

---

## Usage — CLI Modes

```bash
# Binary classifier (focus vs relax)
python main.py --mode train           # train binary model
python main.py --mode evaluate        # evaluate binary model
python main.py --mode predict         # binary prediction demo

# 5-class classifier
python main.py --mode train-multi     # train 5-class model
python main.py --mode evaluate-multi  # evaluate multiclass model
python main.py --mode predict-multi   # multiclass prediction demo

# Advanced features
python main.py --mode dashboard       # launch live real-time dashboard
python main.py --mode export-onnx     # export multiclass model to ONNX
python main.py --mode api             # start FastAPI REST server

# Run everything
python main.py --mode all
```

---

## API Endpoints

Start the server:

```bash
python main.py --mode api
# or: uvicorn api.main:app --reload
```

| Method | Path                  | Description                              |
|--------|-----------------------|------------------------------------------|
| GET    | `/health`             | Health check and model load status       |
| POST   | `/predict`            | Binary prediction (FOCUS / RELAX)        |
| POST   | `/predict/multiclass` | 5-class prediction with probabilities    |
| GET    | `/states`             | All brain states with descriptions       |
| GET    | `/docs`               | Swagger UI (http://localhost:8000/docs)  |

---

## Hardware Setup

### Muse Headset (via LSL)

1. Install `muse-lsl`: `pip install muse-lsl`
2. Pair Muse via Bluetooth.
3. Start the stream: `muselsl stream --address <MUSE_MAC_ADDRESS>`
4. Run the reader: `python hardware/muse_reader.py`

### OpenBCI Cyton / Ganglion (via brainflow)

1. Install brainflow: `pip install brainflow`
2. Connect the USB dongle and power on the board.
3. Run the reader: `python hardware/openbci_reader.py`

Both modules fall back gracefully to **simulated data** if no device is found.

---

## Node.js CLI

```bash
cd nodejs_cli
npm install

# Predict brain state (requires exported ONNX model)
node bin/bci.js predict

# Print model info
node bin/bci.js info
```

See [nodejs_cli/README.md](nodejs_cli/README.md) for details.

---

## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Language           | Python 3.8+                         |
| EEG Simulation     | NumPy                               |
| Signal Filtering   | SciPy (`butter`, `sosfiltfilt`)     |
| Feature Scaling    | scikit-learn `StandardScaler`       |
| Neural Network     | TensorFlow / Keras                  |
| Visualisation      | Matplotlib                          |
| Model Persistence  | Keras `.h5` + `joblib`              |
| ONNX Export        | tf2onnx                             |
| Muse Hardware      | pylsl (Lab Streaming Layer)         |
| OpenBCI Hardware   | brainflow                           |
| REST API           | FastAPI + uvicorn + Pydantic        |
| Node.js CLI        | onnxruntime-node, commander, chalk  |

---

## License

This project is released under the MIT License.

