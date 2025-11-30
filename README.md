# FraudGuard v2 🛡️

> **Real-time Financial Fraud Detection System powered by Fine-tuned Llama-3.1-8B with LoRA**

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://fraudguard-swart.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade fraud detection system featuring a fine-tuned LLM for explainable transaction risk assessment, built with FastAPI, Next.js, and modern MLOps practices.

---

## 🎯 Key Highlights

- ✅ **Custom-trained LLM** using Parameter-Efficient Fine-Tuning (LoRA/QLoRA)
- ✅ **Production ML Pipeline** from data generation → training → inference → deployment
- ✅ **Explainable AI** with SHAP values + natural language explanations
- ✅ **Full-stack deployment** (Frontend: Vercel, Backend: Render)
- ✅ **Enterprise features**: Redis caching, rate limiting, structured logging, health checks

---

## 🚀 Live Production Status

**Current Deployment:**
- **Frontend:** [https://fraudguard-swart.vercel.app](https://fraudguard-swart.vercel.app)
- **Backend:** [https://fraudguard-backend-pk3d.onrender.com](https://fraudguard-backend-pk3d.onrender.com)

> **Note on AI Inference:** The system is currently running in **Fallback Mode** (Rule-based + SHAP). The fine-tuned Llama-3 model is trained and ready but requires a GPU instance (RunPod) to be attached for live inference. This is a cost-optimization decision for the current demo.

---

## 🧠 Machine Learning Pipeline

### Model Architecture

**Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`  
**Fine-tuning Method:** Unsloth + QLoRA (4-bit quantization)  
**Inference Engine:** vLLM for optimized GPU serving

### Training Details

| Component | Value |
|-----------|-------|
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 16 |
| **Learning Rate** | 2e-4 |
| **Quantization** | 4-bit (NF4) |
| **Training Steps** | 500 |
| **GPU** | A100 (compatible, not required for inference) |
| **Est. Training Cost** | ~$15-20 on cloud GPU |
| **Est. Training Time** | ~30-60 minutes |

### Datasets

1. **Kaggle Credit Card Fraud** - 284,807 real-world transactions
2. **IEEE-CIS Fraud Detection** - Transaction + identity features
3. **Synthetic Financial QA** - Generated fraud scenarios with explanations

**See:** [`training/FraudGuard_v2_Training.ipynb`](./training/FraudGuard_v2_Training.ipynb) for full training process.

### Performance Status

> **Note:** Model training pipeline is complete. Formal evaluation on held-out test set is pending.

**Training Configuration (Verified):**
```
LoRA Rank:           16
LoRA Alpha:          16
Learning Rate:       2e-4
Training Steps:      500
Quantization:        4-bit (NF4)
Optimizer:           AdamW 8-bit
```

**Planned Evaluation:**
- Benchmark on IEEE-CIS Fraud Detection test set
- Measure F1, Precision, Recall, AUC-ROC
- Profile inference latency with vLLM
- Evaluate explainability quality

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Next.js   │─────▶│   FastAPI    │─────▶│    vLLM     │
│  Frontend   │      │   Backend    │      │  (Llama-3)  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │   Redis +   │
                     │  Postgres   │
                     └─────────────┘
```

**Tech Stack:**
- **Frontend:** Next.js 15, TypeScript, Tailwind CSS, shadcn/ui
- **Backend:** FastAPI, Pydantic, SlowAPI (rate limiting)
- **ML:** Unsloth, vLLM, Transformers, PyTorch
- **Infrastructure:** Docker, Redis, PostgreSQL
- **Deployment:** Vercel (Frontend), Render (Backend), RunPod (Inference)

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+
- (Optional) NVIDIA GPU for local inference

### Local Development

1. **Clone & Install**
   ```bash
   git clone https://github.com/your-username/fraudguard.git
   cd FRAUDGUARD
   ```

2. **Start Infrastructure**
   ```bash
   docker-compose up -d db redis
   ```

3. **Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

4. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

5. **Access**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

---

## 📊 Model Training (Optional)

The repository includes a pre-trained LoRA model. To train from scratch:

```bash
cd training
pip install -r requirements.txt
python train.py
```

**Outputs:**
- `lora_model/` - LoRA adapter weights
- Logs with training metrics

For interactive training, see: [`FraudGuard_v2_Training.ipynb`](./training/FraudGuard_v2_Training.ipynb)

---

## 🔌 API Endpoints

### `POST /predict`
Predict fraud risk for a transaction.

**Request:**
```json
{
  "transaction_id": "TXN_12345",
  "user_id": "USER_001",
  "amount": 1500.00,
  "merchant": "Unknown Electronics Store",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response:**
```json
{
  "transaction_id": "TXN_12345",
  "is_fraud": true,
  "risk_score": 0.87,
  "explanation": "HIGH RISK: Large transaction ($1500) at unfamiliar merchant. Unusual for user pattern.",
  "latency_ms": 118.3,
  "request_id": "uuid-..."
}
```

### `POST /explain`
Get detailed SHAP-based explanation.

**Response** includes:
- `shap_values` - Feature contributions
- `natural_language_explanation` - LLM-generated reasoning
- `feature_importance` - Normalized importance scores

### `GET /health`
Health check with model status.

---

## 🎨 Features

### Frontend
- ✅ Real-time fraud monitoring dashboard
- ✅ Transaction simulation with live risk scores
- ✅ AI-generated explanations displayed inline
- ✅ System status monitoring (Backend + vLLM connectivity)

### Backend
- ✅ **Explainable Predictions**: SHAP + natural language
- ✅ **Caching**: Redis for repeated transaction queries
- ✅ **Rate Limiting**: 60 req/min default (configurable)
- ✅ **Fallback Mode**: Continues without vLLM (rule-based predictions)
- ✅ **Structured Logging**: JSON logs with request tracing
- ✅ **CORS**: Configurable origins
- ✅ **API Key Auth**: Optional authentication

### ML Inference
- ✅ **vLLM**: Optimized for throughput + low latency
- ✅ **Batching**: Automatic request batching
- ✅ **Quantization**: 4-bit for memory efficiency

---

## 📦 Deployment

### Frontend (Vercel)
```bash
cd frontend
vercel --prod
```

### Backend (Render)
Uses `backend/render.yaml` for automatic deployment.

**Environment Variables:**
```
VLLM_SERVER_URL=https://your-runpod-url/v1
REDIS_URL=redis://your-redis-url:6379
CORS_ORIGINS=https://your-frontend.vercel.app
```

### Inference (RunPod / GPU Platform)
Deploy `inference/Dockerfile` with:
```
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

Or use your fine-tuned model from Hugging Face.

---

## 🧪 Testing

**Backend health check:**
```bash
curl http://localhost:8000/health
```

**Submit test transaction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TEST_001",
    "user_id": "USER_1",
    "amount": 5000,
    "merchant": "Luxury Watches",
    "timestamp": "2024-01-15T10:30:00Z"
  }'
```

---

## 📁 Project Structure

```
FRAUDGUARD/
├── backend/               # FastAPI application
│   ├── main.py           # API routes + ML inference logic
│   ├── feature_store/    # Feast feature engineering
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/              # Next.js dashboard
│   ├── src/
│   │   ├── app/          # App router pages
│   │   └── components/   # UI components (shadcn)
│   └── package.json
├── training/              # ML training pipeline
│   ├── train.py          # LoRA fine-tuning script
│   ├── merge_lora.py     # Merge LoRA with base model
│   ├── lora_model/       # Trained LoRA adapters (output)
│   └── FraudGuard_v2_Training.ipynb
├── inference/             # vLLM inference server
│   ├── Dockerfile        # GPU-optimized container
│   └── start_vllm.sh     # Server startup script
├── data/                  # Datasets + generators
│   ├── generate_synthetic.py
│   └── download_data.py
├── docker-compose.yml     # Local dev infrastructure
└── README.md
```

---

## 🔐 Security Features

- ✅ Input validation with Pydantic
- ✅ SQL injection protection (parameterized queries)
- ✅ XSS protection (input sanitization)
- ✅ CORS whitelist
- ✅ Optional API key authentication
- ✅ Rate limiting

See: [`SECURITY.md`](./SECURITY.md) for details.

---

## 📺 Demo (Coming Soon)

> A Loom walkthrough will be added here showcasing:
> - Live fraud predictions with RunPod GPU inference
> - Training notebook walkthrough
> - Production deployment process

---

## 🛠️ Tech Stack Deep Dive

**Why Llama-3.1-8B?**
- Strong reasoning capabilities for fraud explanation
- Efficient 4-bit quantization (fits on consumer GPUs)
- Commercially licensed

**Why Unsloth + LoRA?**
- 2x faster training than HuggingFace Trainer
- 60% less memory (enables 4-bit + LoRA)
- Fully compatible with vLLM for inference

**Why vLLM?**
- PagedAttention for 24x higher throughput
- Continuous batching
- OpenAI-compatible API

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Training custom LLMs with PEFT (LoRA/QLoRA)
- ✅ Production ML deployment (GPU inference, caching, fallbacks)
- ✅ Full-stack development (React + FastAPI)
- ✅ MLOps (Docker, CI/CD, monitoring, logging)
- ✅ Explainable AI (SHAP + LLM explanations)

---

## 📄 License

MIT - See [LICENSE](LICENSE)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Submit a PR with clear description

---


## ⭐ Acknowledgments

- **Unsloth** - Fast LLM training
- **vLLM** - Efficient inference
- **Hugging Face** - Model hosting
- **shadcn/ui** - Beautiful components
