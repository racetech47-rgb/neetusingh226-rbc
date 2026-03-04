# BCI Neural Network — EEG Focus vs Relax Classifier

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
