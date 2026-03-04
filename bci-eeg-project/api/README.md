# BCI FastAPI REST API

A FastAPI REST service for classifying brain states from EEG data.

## Start the Server

```bash
# From the bci-eeg-project/ directory
pip install -r api/requirements.txt
uvicorn api.main:app --reload

# Or via main.py
python main.py --mode api
```

The API will be available at **http://localhost:8000**.

## Swagger UI

Interactive documentation is automatically available at:

- **http://localhost:8000/docs** — Swagger UI
- **http://localhost:8000/redoc** — ReDoc

## Endpoints

| Method | Path                   | Description                                   |
|--------|------------------------|-----------------------------------------------|
| GET    | `/health`              | Health check — returns model load status      |
| POST   | `/predict`             | Binary classification (FOCUS / RELAX)         |
| POST   | `/predict/multiclass`  | 5-class classification with all probabilities |
| GET    | `/states`              | List all brain states with descriptions       |
| GET    | `/docs`                | Swagger UI (auto-generated)                   |

## Example `curl` Commands

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"loaded"}
```

### Binary Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "eeg_data": [
      [
        [0.1, -0.2, 0.3, 0.05, -0.1, 0.4, -0.3, 0.2],
        [0.2, 0.1, -0.1, 0.3, 0.05, -0.2, 0.1, 0.4]
      ]
    ]
  }'
# {"state":"FOCUS","confidence":0.943}
```

### Multiclass Prediction

```bash
curl -X POST http://localhost:8000/predict/multiclass \
  -H "Content-Type: application/json" \
  -d '{
    "eeg_data": [
      [
        [0.1, -0.2, 0.3, 0.05, -0.1, 0.4, -0.3, 0.2],
        [0.2, 0.1, -0.1, 0.3, 0.05, -0.2, 0.1, 0.4]
      ]
    ]
  }'
# {"state":"MEDITATION","confidence":0.912,"all_probs":{"focus":0.012,...}}
```

### Brain States List

```bash
curl http://localhost:8000/states
```

## Prerequisites

Train the models before starting the API:

```bash
python main.py --mode train        # binary model
python main.py --mode train-multi  # multiclass model
```

## Architecture

```
api/
├── main.py          ← FastAPI app + endpoints
├── schemas.py       ← Pydantic request/response models
├── model_loader.py  ← Singleton model loader (loaded on startup)
├── requirements.txt
└── README.md
```
