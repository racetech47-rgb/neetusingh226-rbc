# BCI Neural Network — EEG Brain-State Classifier (Phase 3)

A Brain-Computer Interface (BCI) project that classifies brain states using
simulated and real EEG (Electroencephalography) data, a deep neural network
built with TensorFlow/Keras, a real-time FastAPI WebSocket backend, and a
React live dashboard.

---

## What is BCI / EEG?

**EEG** records electrical activity produced by neurons in the brain using
electrodes placed on the scalp.  Different mental states are associated with
distinct frequency bands:

| Band   | Frequency | Mental state                         |
|--------|-----------|--------------------------------------|
| Theta  | 4–8 Hz    | Drowsiness, light sleep              |
| Alpha  | 8–13 Hz   | Relaxed, eyes closed, idle           |
| Beta   | 13–30 Hz  | Active thinking, focus, alertness    |

A **Brain-Computer Interface** uses these signals to enable direct
communication between the brain and external devices — without any muscle
movement.

---

## Project Structure

```
bci-eeg-project/
├── README.md
├── requirements.txt
├── main.py                        ← CLI entry point (11 modes)
├── Dockerfile                     ← Docker image for the API
├── docker-compose.yml             ← Full-stack local deployment
├── render.yaml                    ← Render.com deploy config
├── api/
│   └── main.py                    ← FastAPI + WebSocket endpoint
├── data/
│   ├── simulate_eeg.py            ← Binary focus/relax simulator
│   └── simulate_motor_eeg.py      ← 4-class motor imagery simulator
├── preprocessing/
│   └── filter.py                  ← Bandpass filter + FFT features
├── model/
│   ├── train.py                   ← Binary classifier training
│   ├── train_motor.py             ← Motor imagery classifier training
│   ├── evaluate.py                ← Evaluation & confusion matrix
│   ├── finetune.py                ← PhysioNet fine-tuning
│   └── saved_model/               ← Persisted model artefacts
├── inference/
│   ├── predict.py                 ← Binary brain-state prediction
│   └── predict_motor.py           ← Motor imagery prediction
├── datasets/
│   ├── physionet_loader.py        ← PhysioNet dataset downloader
│   └── README.md
├── notebooks/
│   └── physionet_analysis.ipynb   ← EEG analysis notebook
├── control/
│   └── keyboard_control.py        ← Motor → keyboard translation
├── games/
│   └── bci_pong.py                ← Brain-controlled Pong game
├── tests/
│   ├── test_simulate.py
│   ├── test_preprocessing.py
│   └── test_api.py
└── deploy/
    └── README.md                  ← Cloud deployment guide

webapp/                            ← React live dashboard
├── package.json
├── Dockerfile
├── public/index.html
└── src/
    ├── App.jsx
    ├── index.jsx
    └── components/
        ├── BrainStateDisplay.jsx
        ├── EEGChart.jsx
        └── ProbabilityBars.jsx
```

---

## CLI Modes

| Mode | Command | Description |
|------|---------|-------------|
| train | `python main.py --mode train` | Generate data, extract features, train binary classifier |
| evaluate | `python main.py --mode evaluate` | Accuracy, confusion matrix, plots |
| predict | `python main.py --mode predict` | Live binary prediction demo |
| webapp | `python main.py --mode webapp` | Start FastAPI API server |
| docker | `python main.py --mode docker` | Print docker-compose instructions |
| download-data | `python main.py --mode download-data` | Download PhysioNet EEG dataset |
| finetune | `python main.py --mode finetune` | Fine-tune on PhysioNet data |
| train-motor | `python main.py --mode train-motor` | Train 4-class motor imagery model |
| predict-motor | `python main.py --mode predict-motor` | Run motor imagery prediction demo |
| bci-pong | `python main.py --mode bci-pong` | Launch brain-controlled Pong game |
| all | `python main.py --mode all` | Run train → evaluate → predict |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       React Web Dashboard                        │
│   BrainStateDisplay │ EEGChart │ ProbabilityBars                 │
└───────────────────────────┬─────────────────────────────────────┘
                WebSocket (ws://localhost:8000/ws)
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Backend (api/main.py)               │
│  GET /health │ GET /states │ POST /predict │ WS /ws             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         │                                     │
┌────────▼────────┐                  ┌─────────▼────────┐
│  Binary Model   │                  │  Motor Model     │
│  (focus/relax)  │                  │  (4-class LRFR)  │
└────────┬────────┘                  └─────────┬────────┘
         │                                     │
┌────────▼────────────────────────────────────▼────────┐
│              Preprocessing Pipeline                   │
│  bandpass_filter → FFT band-power → StandardScaler    │
└───────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│              EEG Data Sources                          │
│  simulate_eeg.py │ simulate_motor_eeg.py │ PhysioNet  │
└───────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/racetech47-rgb/neetusingh226-rbc.git
cd neetusingh226-rbc/bci-eeg-project

# Create a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For the React webapp (Node.js 18+ required):

```bash
cd ../webapp
npm install
```

---

## Web UI

The React dashboard connects via WebSocket to stream real-time predictions:

```bash
# Terminal 1: Start the API
cd bci-eeg-project
python main.py --mode webapp

# Terminal 2: Start the React app
cd webapp
npm start
```

Visit `http://localhost:3000` to see the live dashboard.

**Brain state colours:**

| State | Colour | Emoji |
|-------|--------|-------|
| FOCUS | 🔵 Blue | 🎯 |
| RELAX | 🟢 Green | 😌 |
| STRESS | 🔴 Red | 😰 |
| SLEEP | 🟣 Purple | 😴 |
| MEDITATION | 🟡 Yellow | 🧘 |

---

## Cloud Deployment

See [deploy/README.md](deploy/README.md) for full instructions.

### Docker (local)

```bash
docker-compose up --build
```

### Railway / Render

```bash
# Railway
railway up

# Render: connect repo in dashboard — render.yaml is auto-detected
```

---

## Motor Imagery + BCI Pong

Train the motor imagery classifier and launch the game:

```bash
python main.py --mode train-motor
python main.py --mode bci-pong
```

**Motor imagery classes:**

| Class | Action |
|-------|--------|
| LEFT HAND | Move cursor left / Turn left |
| RIGHT HAND | Move cursor right / Turn right |
| FEET | Move forward / Scroll down |
| REST | Stop / No action |

---

## Real EEG Dataset (PhysioNet)

```bash
# Download subjects 1-3
python main.py --mode download-data

# Fine-tune the model on real EEG
python main.py --mode finetune
```

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Neural Network | TensorFlow / Keras |
| EEG Simulation | NumPy |
| Signal Filtering | SciPy |
| Feature Scaling | scikit-learn |
| REST API | FastAPI |
| WebSocket | FastAPI + websockets |
| React UI | React 18 + recharts |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Cloud Deploy | Railway / Render |
| Real EEG Data | MNE + PhysioNet |
| Keyboard Control | pyautogui |
| BCI Game | pygame |
| Tests | pytest + httpx |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is released under the MIT License.

A Brain-Computer Interface (BCI) project that classifies brain states as
**Focus** or **Relax** using simulated EEG (Electroencephalography) data and a
deep neural network built with TensorFlow/Keras.

---

## What is BCI / EEG?

**EEG** records electrical activity produced by neurons in the brain using
electrodes placed on the scalp.  Different mental states are associated with
distinct frequency bands:

| Band   | Frequency | Mental state                         |
|--------|-----------|--------------------------------------|
| Theta  | 4–8 Hz    | Drowsiness, light sleep              |
| Alpha  | 8–13 Hz   | Relaxed, eyes closed, idle           |
| Beta   | 13–30 Hz  | Active thinking, focus, alertness    |

A **Brain-Computer Interface** uses these signals to enable direct
communication between the brain and external devices — without any muscle
movement.

This project **simulates** EEG data for two classes (focus / relax),
extracts spectral features, and trains a binary classifier.

---

## Project Structure

```
bci-eeg-project/
├── README.md
├── requirements.txt
├── main.py                        ← CLI entry point
├── data/
│   └── simulate_eeg.py            ← Simulated EEG signal generator
├── preprocessing/
│   ├── __init__.py
│   └── filter.py                  ← Bandpass filter + FFT + feature extraction
├── model/
│   ├── __init__.py
│   ├── train.py                   ← Neural network training (TensorFlow/Keras)
│   ├── evaluate.py                ← Accuracy, confusion matrix, plots
│   └── saved_model/               ← Persisted model artefacts (auto-created)
│       ├── bci_model.h5
│       ├── scaler.pkl
│       └── history.npy
└── inference/
    ├── __init__.py
    └── predict.py                 ← Real-time classification
```

---

## Architecture

```
EEG Signal (8 ch × 512 pts)
         │
         ▼
┌────────────────────┐
│  Bandpass Filter   │   theta / alpha / beta bands per channel
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
         ▼
┌────────────────────────────────┐
│  Dense(128, relu) + Dropout    │
│  Dense(64,  relu) + Dropout    │
│  Dense(1, sigmoid)             │
└────────┬───────────────────────┘
         │
         ▼
  "FOCUS" / "RELAX"
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/racetech47-rgb/neetusingh226-rbc.git
cd neetusingh226-rbc/bci-eeg-project

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Train the model (generates data, extracts features, trains & saves model)
python main.py --mode train

# Evaluate the saved model (classification report + plots)
python main.py --mode evaluate

# Run a real-time prediction demo
python main.py --mode predict

# Run all three steps in sequence
python main.py --mode all
```

---

## Example Output

### Train
```
[1/4] Generating simulated EEG data …
      signals shape: (1000, 8, 512)  labels shape: (1000,)
[2/4] Extracting features …
      features shape: (1000, 24)
[3/4] Splitting data into train/test sets …
      train: 800 samples  |  test: 200 samples
[4/4] Building and training model …
…
✅ Training accuracy   : 98.50%
✅ Validation accuracy : 97.00%
```

### Evaluate
```
=== Classification Report ===
              precision    recall  f1-score   support

       Relax     0.9850    0.9700    0.9774       100
       Focus     0.9703    0.9850    0.9776       100

    accuracy                         0.9775       200
```

### Predict
```
🔬 Running real-time prediction demo (5 samples) …

  [✓] 🧠 Brain State: FOCUS (confidence:  94.3%)  [true: FOCUS]
  [✓] 🧠 Brain State: RELAX (confidence:  97.1%)  [true: RELAX]
  [✓] 🧠 Brain State: FOCUS (confidence:  88.6%)  [true: FOCUS]
  [✓] 🧠 Brain State: RELAX (confidence:  91.4%)  [true: RELAX]
  [✓] 🧠 Brain State: FOCUS (confidence:  96.0%)  [true: FOCUS]
```

---

## Tech Stack

| Component          | Technology                |
|--------------------|---------------------------|
| Language           | Python 3.8+               |
| EEG Simulation     | NumPy                     |
| Signal Filtering   | SciPy (`butter`, `sosfiltfilt`) |
| Feature Scaling    | scikit-learn `StandardScaler` |
| Neural Network     | TensorFlow / Keras        |
| Visualisation      | Matplotlib                |
| Model Persistence  | Keras `.h5` + `joblib`    |

---

## License

This project is released under the MIT License.
