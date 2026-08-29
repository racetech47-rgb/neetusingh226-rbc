# BCI API — Cloud Deployment Guide

This guide covers three deployment options: **Railway**, **Render**, and
**Docker (local)**.

---

## 1. Docker (Local)

The fastest way to run the full stack locally.

```bash
cd bci-eeg-project
docker-compose up --build
```

| Service | URL |
|---------|-----|
| BCI API | http://localhost:8000 |
| Web Dashboard | http://localhost:3000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## 2. Railway Deployment

Railway offers a free tier and supports Docker deployments.

### Prerequisites

- [Railway account](https://railway.app)
- [Railway CLI](https://docs.railway.app/develop/cli) — `npm install -g @railway/cli`

### Steps

```bash
# 1. Authenticate
railway login

# 2. Initialise a new project (run from bci-eeg-project/)
railway init

# 3. Deploy
railway up

# 4. Set environment variables in the Railway dashboard
#    PROJECT → Settings → Variables:
#      MODEL_PATH = model/saved_model/bci_model.h5
```

### GitHub Actions Auto-Deploy

Set the following secrets in your GitHub repository
(Settings → Secrets → Actions):

| Secret | Value |
|--------|-------|
| `RAILWAY_TOKEN` | Your Railway project token |

The included `deploy.yml` workflow will automatically deploy on every push to
`main`.

---

## 3. Render Deployment

Render has a free tier for web services.

### Steps

1. Go to [render.com](https://render.com) and create a new account.
2. Click **New → Web Service** and connect your GitHub repository.
3. Set the **Root Directory** to `bci-eeg-project`.
4. Render will auto-detect the `render.yaml` file and configure the service.
5. Click **Create Web Service**.

Render will build and deploy automatically on every push to `main`.

### Environment Variables

Set in the Render dashboard under **Environment**:

| Variable | Value |
|----------|-------|
| `PYTHON_VERSION` | `3.10.0` |

---

## 4. GitHub Actions Secrets Required

| Secret | Purpose |
|--------|---------|
| `RAILWAY_TOKEN` | Authenticate Railway CLI deployments |

To add secrets:
1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add each secret listed above
