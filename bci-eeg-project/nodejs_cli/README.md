# BCI CLI — Node.js ONNX Inference

A Node.js CLI tool that loads the exported ONNX BCI model and classifies
brain states directly from the command line.

## Prerequisites

- Node.js ≥ 18
- The ONNX model must be exported first (see below)

## Install

```bash
# From the nodejs_cli/ directory
npm install
```

## Export the ONNX Model

Before running the CLI you must train the multiclass model and export it:

```bash
# From bci-eeg-project/
pip install -r requirements.txt

python main.py --mode train-multi   # train 5-class model
python main.py --mode export-onnx   # export to export/bci_model.onnx
```

## Usage

```bash
# Predict brain state from simulated EEG (1 sample)
node bin/bci.js predict

# Predict 5 samples
node bin/bci.js predict --samples 5

# Print model info and brain state descriptions
node bin/bci.js info
```

## Example Output

```
🧠 BCI Brain State Classifier — predicting 1 sample(s)…

  [1] 🧠 Brain State: MEDITATION  (confidence: 91.2%)
```

## Commands

| Command              | Description                                      |
|----------------------|--------------------------------------------------|
| `bci predict`        | Run inference on simulated EEG, print brain state |
| `bci predict -n <N>` | Classify N simulated samples                     |
| `bci info`           | Print model input/output shapes and class list   |

## Architecture

```
nodejs_cli/
├── bin/
│   └── bci.js          ← CLI entry point (commander)
├── src/
│   └── inference.js    ← ONNX inference module (onnxruntime-node)
├── package.json
└── README.md
```

The CLI reads the ONNX model from `../export/bci_model.onnx` relative to the
`nodejs_cli/` directory.
