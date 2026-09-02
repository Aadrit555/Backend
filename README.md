# Unified AI Platform

A powerful, native hardware-accelerated AI platform for orchestrating RAG pipelines, fine-tuning LLMs via Unsloth, and running advanced computer vision tasks directly on your local GPU.

## Prerequisites

- **OS:** Linux (Ubuntu/Debian recommended) or WSL2
- **GPU:** NVIDIA GPU with CUDA support (for Unsloth fine-tuning)
- **Python:** 3.10+
- **Node.js:** v18+ (for local frontend development)

---

## 🚀 Backend Setup

The backend is built with FastAPI and runs natively on your machine to fully utilize local GPU acceleration for model training and deployment.

### 1. Initialize the Environment
Open a terminal and navigate to the project directory:
```bash
cd unified
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required machine learning and server dependencies:
```bash
pip install -r requirements.txt
```
*(Note: If you run into `bitsandbytes` or `xformers` issues, ensure your CUDA drivers are up to date!)*

### 3. Environment Variables
Create a `.env` file in the `backend/` directory:
```bash
cp backend/.env.example backend/.env
```
Inside the `backend/.env` file, populate it with your keys like this:

```env
# GROQ_API_KEY_1 is required. _2 through _5 are optional — used for
# automatic failover when a key hits its daily/rate limit.
GROQ_API_KEY_1=
GROQ_API_KEY_2=
GROQ_API_KEY_3=
GROQ_API_KEY_4=
GROQ_API_KEY_5=

# Groq Model
GROQ_MODEL=openai/gpt-oss-120b

# OpenRouter API Key for Cloud Model Fallbacks
OPENROUTER_API_KEY=
```

### 4. Start the Server
Start the FastAPI server:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is now running at `http://localhost:8000`.

---

## 🌐 Using the UI

You don't need to run or deploy a local frontend! We provide an officially hosted UI that securely connects directly to your local backend from your browser. 

Once your backend is running locally, simply visit:
**👉 [https://unified-model-interface.vercel.app/](https://unified-model-interface.vercel.app/)**

Your local GPU will power all requests (data ingestion, RAG querying, and model training) while you control it seamlessly from the web interface.
---

## 📁 Repository Structure

- `/backend`: FastAPI server, ML adapters (Unsloth, RAG, AutoGluon), and SQLite database.
- `/frontend`: Next.js frontend application (React, TailwindCSS).
- `/backend/storage`: (Git-ignored) Local directory for raw datasets, model artifacts, and vector indices.
