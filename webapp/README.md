# BCI Web Dashboard

React-based real-time brain-state visualisation dashboard.

## Prerequisites

- Node.js 18+
- The BCI FastAPI backend running locally

## Quick Start

### 1. Start the API backend

```bash
cd bci-eeg-project
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000` and the WebSocket endpoint
at `ws://localhost:8000/ws`.

### 2. Start the React webapp

```bash
cd webapp
npm install
npm start
```

The dashboard will open at `http://localhost:3000`.

## Features

| Feature | Description |
|---------|-------------|
| 🧠 Brain State Badge | Animated display of current brain state with emoji |
| 📊 Probability Bars | Real-time probability for each of 5 brain states |
| 📈 EEG Waveform | Live scrolling chart of all 8 EEG channels |
| 🔄 Auto-reconnect | Automatically reconnects to WebSocket on disconnect |

## Brain States

| State | Colour | Emoji |
|-------|--------|-------|
| FOCUS | 🔵 Blue | 🎯 |
| RELAX | 🟢 Green | 😌 |
| STRESS | 🔴 Red | 😰 |
| SLEEP | 🟣 Purple | 😴 |
| MEDITATION | 🟡 Yellow | 🧘 |

## Docker

```bash
# Build and run with docker-compose from bci-eeg-project/
docker-compose up
```

The webapp will be available at `http://localhost:3000`.
