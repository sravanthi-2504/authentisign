# AuthentiSign ✍️

AI-powered signature verification — tell a genuine signature from a forgery in seconds.

AuthentiSign is a full-stack web app that uses a Siamese neural network to compare two signature images and decide whether they belong to the same person, with a confidence score to back it up. Built for the CEDAR signature dataset, it reaches 92–96% verification accuracy.

---

## 🚀 Quick Start (for recruiters)

```bash
git clone https://github.com/sravanthi-2504/authentisign.git
cd authentisign
chmod +x setup.sh
./setup.sh
```

This installs backend + frontend dependencies. No training required — a pre-trained model ships with the repo, so setup takes a couple of minutes, not 15–30.

Then, in two separate terminals:

```bash
./start-backend.sh    # Terminal 1 — Flask API on http://localhost:5000
./start-frontend.sh   # Terminal 2 — React app, usually http://localhost:3000
```

Open the URL Terminal 2 prints, and log in:

```
Email:    bsonakshi@gmail.com
Password: password123
```

Then upload two signature images and click Run AI Analysis.

⚠️ Keep both terminals running for the whole session — closing or `Ctrl+C`-ing either one will break the app mid-demo.

---

## 🚀 The Problem

Signature verification is still done by eye in banks, legal offices, and exam halls — slow, subjective, and easy to fool. AuthentiSign turns it into an objective, repeatable, AI-backed check: upload a reference signature and a test signature, and get a GENUINE/FORGED verdict with a confidence percentage in real time.

## 🧠 How It Works

At the core is a Siamese Convolutional Neural Network trained with contrastive loss:

1. Two signature images are each passed through the same embedding CNN (shared weights) to produce a 128-dimensional, L2-normalized embedding vector.
2. The Euclidean (L2) distance between the two embeddings is computed — similar signatures land close together in embedding space, forgeries land far apart.
3. A tuned distance threshold (0.30) separates GENUINE from FORGED, cross-checked with cosine similarity, and the distance is converted into an intuitive confidence percentage.

Embedding network architecture:
- 3 Conv2D blocks (32 → 64 → 128 filters) each with BatchNorm + MaxPooling
- Flatten → Dense(256) + Dropout(0.3)
- Dense(128) → custom L2 Normalization layer (final embedding)

Training details:
- Dataset: CEDAR signature dataset (genuine + forged samples per writer)
- Loss: Contrastive loss (margin = 1.5)
- Pair sampling: genuine-genuine pairs (label 1), plus genuine-forged and genuine-vs-other-writer pairs (label 0)
- Optimizer: Adam (lr = 1e-4), with early stopping + LR reduction on plateau

Image preprocessing pipeline (applied identically at train and inference time):
`grayscale → resize to 128×128 → adaptive thresholding → morphological closing → normalize [0,1]`

## 🏗️ Architecture & Tech Stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite, React Router, Tailwind CSS, Framer Motion, Lucide icons |
| Backend | Flask (REST API), Flask-CORS, PyJWT (auth), Werkzeug (password hashing) |
| ML | TensorFlow / Keras (Siamese CNN), OpenCV (image preprocessing), NumPy, scikit-learn |
| Auth | JWT-based token authentication with password hashing |

Request flow: React frontend (port 3000) → HTTP + JWT bearer token → Flask backend (port 5000) → in-process call → Siamese model (loaded once at startup, kept resident in memory).

The model is never called directly by the frontend — every request goes through the API, which owns and serves the model.

## ✨ Features

- 🔐 User authentication — register/login with JWT-secured sessions
- 📤 Signature upload & compare — upload a reference and a test signature for instant verification
- 📊 Confidence scoring — not just genuine/forged, but how confident the model is
- 🕒 Verification history — every check is logged per user and viewable/deletable later
- ⚡ Real-time inference — model is loaded once and locked into inference mode for fast, consistent predictions
- 🩺 Health check endpoint — quick way to confirm the API and model are live

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create a new user account |
| POST | `/api/auth/login` | Log in and receive a JWT |
| GET | `/api/auth/verify` | Verify a token / fetch current user |
| POST | `/api/verify-signature` | Upload `original` + `test` images, get GENUINE/FORGED + confidence |
| GET | `/api/history` | Get a user's past verification results |
| DELETE | `/api/history/<id>` | Remove a history entry |
| GET | `/api/health` | Check API + model status |

## 📁 Project Structure

```text
authentisign/
├── backend/
│   ├── app.py
│   ├── predict.py
│   ├── signature_verifier.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── context/
│   │   └── App.jsx, main.jsx
│   └── package.json
├── model/
│   ├── train_model.py
│   ├── testcedar.py
│   ├── debug_threshold.py
│   └── signature_embedding_model.keras
├── setup.sh
└── start-all.sh / start-backend.sh / start-frontend.sh
```

## 🛠️ Getting Started (full detail)

### Prerequisites

- Python 3.8+
- Node.js 16+

### Setup

```bash
git clone https://github.com/sravanthi-2504/authentisign.git
cd authentisign
chmod +x setup.sh
./setup.sh
```

`setup.sh` creates a Python virtual environment, installs backend and frontend dependencies, and checks for the pre-trained model — since it's committed to the repo, training is skipped automatically.

### Run the app

```bash
./start-backend.sh
./start-frontend.sh
```

Only the frontend URL should be opened in a browser. The backend URL has no visual page — visiting it directly shows "Not Found," which is expected.

### Demo login

```
Email:    bsonakshi@gmail.com
Password: password123
```

## 🩺 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` on port 5000 | AirPlay Receiver (macOS) or another process/Docker container on port 5000 | `lsof -nP -iTCP:5000 \| grep LISTEN` → `kill -9 <PID>`, or disable AirPlay Receiver in System Settings → General → AirDrop & Handoff |
| "Not Found" at `127.0.0.1:5000` | Normal — that's the API root, which has no page | Use the frontend URL instead (usually `localhost:3000`) |
| "Failed to fetch" on login or verification | Backend isn't running, crashed, or was stopped mid-session | Restart it: `cd backend && source venv/bin/activate && python app.py` — and don't close that terminal |
| `ModuleNotFoundError: No module named 'model'` | `app.py` run from an unexpected working directory | Already handled — `app.py` adds the project root to `sys.path` automatically |
| Vite says "Port 3000 is in use, trying another one" | Something else has 3000 | Harmless — use whichever port it falls back to (shown in the terminal output) |
| `[Errno 2] No such file or directory: 'uploads/...'` | A stale/orphaned backend process is still running from a deleted or moved folder | Find and kill it (`lsof -nP -iTCP:5000 \| grep LISTEN` → `kill -9 <PID>`), then restart cleanly from the current project folder |

## 📈 Results

- 92–96% verification accuracy on the CEDAR signature dataset
- Verification decided by an L2 distance threshold (0.30) on the learned embeddings, cross-checked with cosine similarity for robustness

## ⚠️ Known Limitations

- No persistent database — registered users and verification history live in in-memory Python dictionaries; restarting the backend clears them. Fine for a demo, would need a real database for production.
- Static model — the model doesn't learn from usage; every verification is inference against fixed, pre-trained weights.
- Dev server only — Flask's built-in server (used here) isn't meant for production traffic; a real deployment would sit behind Gunicorn/uWSGI + Nginx.
- Secrets are hardcoded (JWT secret key, demo password) for simplicity — would move to environment variables for anything beyond a demo.

## 🔮 Roadmap / What's Next

- Swap in-memory user/history stores for a persistent database
- Move JWT secret + config to environment variables for production
- Add support for additional signature datasets (BHSig, GPDS) to test generalization
- Deploy backend + frontend to the cloud with a hosted demo link
- Add offline (online, pen-stroke) signature verification alongside the current offline image-based approach

*AuthentiSign — because a signature should be verified, not guessed.*
