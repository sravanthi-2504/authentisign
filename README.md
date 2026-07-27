# AuthentiSign ✍️

**AI-powered signature verification — tell a genuine signature from a forgery in seconds.**

AuthentiSign is a full-stack web app that uses a Siamese neural network to compare two signature images and decide whether they belong to the same person, with a confidence score to back it up. Built for the CEDAR signature dataset, it reaches **92–96% verification accuracy**.

---

## 🚀 The Problem

Signature verification is still done by eye in banks, legal offices, and exam halls — slow, subjective, and easy to fool. AuthentiSign turns it into an objective, repeatable, AI-backed check: upload a reference signature and a test signature, and get a GENUINE/FORGED verdict with a confidence percentage in real time.

## 🧠 How It Works

At the core is a **Siamese Convolutional Neural Network** trained with contrastive loss:

1. Two signature images are each passed through the *same* embedding CNN (shared weights) to produce a 128-dimensional, L2-normalized embedding vector.
2. The **Euclidean (L2) distance** between the two embeddings is computed — similar signatures land close together in embedding space, forgeries land far apart.
3. A tuned distance threshold separates GENUINE from FORGED, and the distance is converted into an intuitive confidence percentage.

**Embedding network architecture:**
- 3 Conv2D blocks (32 → 64 → 128 filters) each with BatchNorm + MaxPooling
- Flatten → Dense(256) + Dropout(0.3)
- Dense(128) → custom L2 Normalization layer (final embedding)

**Training details:**
- Dataset: CEDAR signature dataset (genuine + forged samples per writer)
- Loss: Contrastive loss (margin = 1.5)
- Pair sampling: genuine-genuine pairs (label 1), plus genuine-forged and genuine-vs-other-writer pairs (label 0)
- Optimizer: Adam (lr = 1e-4), with early stopping + LR reduction on plateau

**Image preprocessing pipeline** (applied identically at train and inference time):
`grayscale → resize to 128×128 → adaptive thresholding → morphological closing → normalize [0,1]`

## 🏗️ Architecture & Tech Stack

| Layer | Tech |
|---|---|
| **Frontend** | React 18 + Vite, React Router, Tailwind CSS, Framer Motion, Lucide icons |
| **Backend** | Flask (REST API), Flask-CORS, PyJWT (auth), Werkzeug (password hashing) |
| **ML** | TensorFlow / Keras (Siamese CNN), OpenCV (image preprocessing), NumPy, scikit-learn |
| **Auth** | JWT-based token authentication with password hashing |

**Flow:** React frontend → REST calls to Flask API → images preprocessed with OpenCV → passed through the trained Keras embedding model → distance + confidence computed → verdict + history stored and returned.

## ✨ Features

- 🔐 **User authentication** — register/login with JWT-secured sessions
- 📤 **Signature upload & compare** — upload a reference and a test signature for instant verification
- 📊 **Confidence scoring** — not just genuine/forged, but *how* confident the model is
- 🕒 **Verification history** — every check is logged per user and viewable/deletable later
- ⚡ **Real-time inference** — model is loaded once and locked into inference mode for fast, consistent predictions
- 🩺 **Health check endpoint** — quick way to confirm the API and model are live

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

```
authentisign/
├── backend/
│   ├── app.py                 # Flask API (auth, verification, history)
│   ├── predict.py             # Standalone prediction helper
│   ├── signature_verifier.py  # Verification logic
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/             # AuthPage, VerifySignature, VerificationHistory, DashboardLayout
│   │   ├── components/        # ProtectedRoute
│   │   └── App.jsx, main.jsx
│   └── package.json
├── model/
│   ├── train_model.py         # Siamese network + training pipeline
│   ├── testcedar.py           # Model evaluation on CEDAR
│   ├── debug_threshold.py     # Threshold tuning
│   └── sample_dataset/
├── setup.sh                   # One-shot install + train + configure script
├── start-all.sh / start-backend.sh / start-frontend.sh
└── check_distances.py         # Sanity-check embedding distances
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+

### Quick setup
```bash
git clone https://github.com/sravanthi-2504/authentisign.git
cd authentisign
chmod +x setup.sh
./setup.sh
```
This installs backend + frontend dependencies and trains the model (~15–30 min on CEDAR).

### Run the app
```bash
./start-all.sh
```
Or run each side separately:
```bash
./start-backend.sh   # Flask API → http://localhost:5000
./start-frontend.sh  # React app → http://localhost:3000
```

### Demo login
```
Email:    bsonakshi@gmail.com
Password: password123
```

## 📈 Results

- **92–96% verification accuracy** on the CEDAR signature dataset
- Verification decided by an L2 distance threshold (0.30) on the learned embeddings, cross-checked with cosine similarity for robustness

## 🔮 Roadmap / What's Next

- Swap in-memory user/history stores for a persistent database
- Move JWT secret + config to environment variables for production
- Add support for additional signature datasets (BHSig, GPDS) to test generalization
- Deploy backend + frontend to the cloud with a hosted demo link
- Add offline (online, pen-stroke) signature verification alongside the current offline image-based approach

## Built by [C Sai Sravanthi](https://github.com/sravanthi-2504) for a research purpose.

---

*AuthentiSign — because a signature should be verified, not guessed.*
