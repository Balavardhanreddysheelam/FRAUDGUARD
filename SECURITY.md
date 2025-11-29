# Security Documentation - FraudGuard

## 🔒 Security Overview

FraudGuard uses a **local 8B model** for fraud detection. This document outlines the security architecture and practices.

## 🎯 Key Security Features

### 1. Local Model Inference
- **No External API Calls**: All fraud detection is performed using a locally hosted Llama-3.1-8B model
- **Model Location**: `inference/model/fraudguard-8b-merged`
- **Privacy**: Transaction data never leaves your infrastructure
- **Compliance**: Full control over data processing and storage

### 2. API Security
- **Optional API Key Authentication**: Configurable via `ENABLE_AUTH` environment variable
- **Rate Limiting**: Default 60 requests/minute per IP (configurable)
- **CORS Protection**: Restricted to specified origins
- **Input Validation**: Comprehensive Pydantic validators for all inputs

### 3. Data Security
- **No Data Storage**: Transaction data is processed in-memory and not persisted
- **Optional Caching**: Redis caching can be enabled (TTL: 1 hour default)
- **No External Dependencies**: Model inference is self-contained

## 🛡️ Security Architecture

```
┌─────────────┐
│   Frontend  │ (Next.js - Vercel)
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────┐
│   Backend   │ (FastAPI - Render)
│  - Auth     │
│  - Rate Lim │
│  - Validate │
└──────┬──────┘
       │ Internal Network
       ▼
┌─────────────┐
│  vLLM Server│ (Local GPU Instance)
│  - Model    │
│  - Inference│
└─────────────┘
```

## 🔐 Environment Variables

### Required for Production
```bash
# API Security
ENABLE_AUTH=true
API_KEY=<strong-random-key>  # Generate with: openssl rand -hex 32

# CORS
CORS_ORIGINS=https://yourdomain.com

# vLLM Server (local or private network)
VLLM_SERVER_URL=http://localhost:8001/v1  # or private URL
```

### Optional
```bash
# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Caching
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
```

## 🚫 What We DON'T Do

- ❌ **No External API Calls**: No calls to OpenAI, Anthropic, or other external services
- ❌ **No Data Transmission**: Transaction data stays within your infrastructure
- ❌ **No Model Upload**: Model files are stored locally, never uploaded
- ❌ **No Third-Party Tracking**: No analytics or tracking services

## ✅ Security Best Practices

### 1. API Key Management
- Generate strong API keys: `openssl rand -hex 32`
- Store in environment variables, never in code
- Rotate keys regularly
- Use different keys for development and production

### 2. Network Security
- Use HTTPS in production (via reverse proxy or load balancer)
- Restrict CORS to specific domains
- Use private networks for vLLM server communication
- Implement firewall rules

### 3. Model Security
- Store model files securely
- Use read-only mounts in Docker
- Restrict access to model directory
- Verify model integrity (checksums)

### 4. Input Validation
- All inputs validated with Pydantic
- Amount limits: 0 < amount <= 10,000,000
- String sanitization for merchant names
- Timestamp format validation

### 5. Error Handling
- No sensitive information in error messages
- Structured logging (no PII in logs)
- Graceful degradation (fallback mode if model unavailable)

## 📊 Security Checklist

### Before Production Deployment
- [ ] `ENABLE_AUTH=true` set
- [ ] Strong `API_KEY` generated and set
- [ ] `CORS_ORIGINS` restricted to production domains
- [ ] HTTPS enabled (via reverse proxy)
- [ ] Rate limiting configured appropriately
- [ ] Model files secured (read-only access)
- [ ] Environment variables not committed to git
- [ ] `.env` file in `.gitignore`
- [ ] Database passwords are strong
- [ ] Logs don't contain sensitive data

## 🔍 Security Monitoring

### Health Checks
- `/health` endpoint verifies model is loaded
- Component status monitoring (vLLM, Redis)
- Automatic fallback if model unavailable

### Logging
- Structured JSON logs
- Request ID tracking
- No PII in logs
- Error tracking without exposing internals

## 🚨 Incident Response

### If Model Fails to Load
1. Check `/health` endpoint
2. Verify model files exist at `inference/model/fraudguard-8b-merged`
3. Check vLLM server logs
4. Service automatically falls back to rule-based prediction

### If API Key Compromised
1. Generate new API key: `openssl rand -hex 32`
2. Update `API_KEY` environment variable
3. Restart service
4. Update all clients with new key

### If Rate Limit Exceeded
- Check `RATE_LIMIT_PER_MINUTE` setting
- Review logs for abuse patterns
- Adjust rate limit if legitimate traffic

## 📝 Compliance Notes

### Data Privacy
- **No Data Storage**: Transactions processed in-memory only
- **No External Sharing**: Data never sent to third parties
- **Local Processing**: All inference happens on your infrastructure

### Model Licensing
- Llama-3.1-8B uses Meta's Llama 3.1 Community License
- Fine-tuned model inherits same license
- Review license terms for commercial use

## 🔗 Related Documentation

- `SECURITY_REVIEW.md` - Complete security audit
- `PRODUCTION_DEPLOYMENT.md` - Deployment security checklist
- `backend/env.example` - Environment variable template

## 📞 Security Contact

For security issues or questions:
1. Review this document
2. Check `SECURITY_REVIEW.md` for detailed audit
3. Follow incident response procedures above

---

**Last Updated**: 2024
**Model**: fraudguard-8b-merged (Llama-3.1-8B)
**Architecture**: Local inference, no external API calls

