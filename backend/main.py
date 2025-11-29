
import os
import time
import logging
import uuid
from typing import Optional, Dict, Tuple, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, HttpUrl
from pydantic_settings import BaseSettings
import httpx
from openai import AsyncOpenAI
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Settings with validation
class Settings(BaseSettings):
    """Application settings with environment variable validation"""
    vllm_server_url: str = Field(default="http://localhost:8001/v1", validation_alias="VLLM_SERVER_URL")
    api_key: Optional[str] = Field(default=None, validation_alias="API_KEY")
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5000", validation_alias="CORS_ORIGINS")
    vllm_timeout: float = Field(default=30.0, validation_alias="VLLM_TIMEOUT")
    cache_ttl: int = Field(default=3600, validation_alias="CACHE_TTL")
    rate_limit_per_minute: int = Field(default=60, validation_alias="RATE_LIMIT_PER_MINUTE")
    enable_auth: bool = Field(default=False, validation_alias="ENABLE_AUTH")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize Redis for caching
redis_client: Optional[redis.Redis] = None

# Initialize vLLM client
vllm_client: Optional[AsyncOpenAI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown"""
    global redis_client, vllm_client
    
    # Startup
    logger.info("Starting FraudGuard API v2")
    
    # Initialize Redis
    try:
        redis_client = await redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching disabled.")
        redis_client = None
    
    # Initialize vLLM client
    try:
        vllm_client = AsyncOpenAI(
            base_url=settings.vllm_server_url,
            api_key="not-needed",
            timeout=httpx.Timeout(settings.vllm_timeout, connect=5.0)
        )
        # Test connection
        await vllm_client.models.list()
        logger.info(f"vLLM client connected to {settings.vllm_server_url}")
    except Exception as e:
        logger.warning(f"vLLM client initialization failed: {e}. Fallback mode enabled.")
        vllm_client = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down FraudGuard API v2")
    if redis_client:
        await redis_client.close()
    if vllm_client:
        await vllm_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard API v2",
    description="Real-time Fraud Detection System with Llama-3.1-8B",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
cors_origins = [origin.strip() for origin in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# API Key authentication (optional)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Verify API key if authentication is enabled"""
    if not settings.enable_auth:
        return True
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return True

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracing"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Pydantic models with validation
class Transaction(BaseModel):
    """Transaction model with comprehensive validation"""
    transaction_id: str = Field(..., min_length=1, max_length=100, description="Unique transaction identifier")
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    amount: float = Field(..., gt=0, le=10000000, description="Transaction amount (must be positive)")
    merchant: str = Field(..., min_length=1, max_length=200, description="Merchant name")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp format"""
        try:
            from datetime import datetime
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Invalid timestamp format. Use ISO 8601 format (e.g., 2024-01-15T10:30:00Z)")
    
    @field_validator("merchant")
    @classmethod
    def sanitize_merchant(cls, v: str) -> str:
        """Sanitize merchant name"""
        # Remove potentially dangerous characters
        return "".join(c for c in v if c.isalnum() or c in " .-_")

class FraudPrediction(BaseModel):
    """Fraud prediction response model"""
    transaction_id: str
    is_fraud: bool
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score between 0 and 1")
    explanation: str
    latency_ms: float
    request_id: Optional[str] = None

class ExplanationRequest(BaseModel):
    """Explanation request model"""
    transaction_id: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0, le=10000000)
    merchant: str = Field(..., min_length=1, max_length=200)
    timestamp: str

class ExplanationResponse(BaseModel):
    """Explanation response model"""
    transaction_id: str
    shap_values: Dict[str, float]
    natural_language_explanation: str
    feature_importance: Dict[str, float]
    request_id: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str
    detail: str
    request_id: Optional[str] = None
    timestamp: str

# Helper functions
def format_transaction_for_llm(transaction: Transaction) -> str:
    """Format transaction data for LLM input"""
    return f"Transaction Amount: ${transaction.amount:.2f}, Merchant: {transaction.merchant}, User ID: {transaction.user_id}, Timestamp: {transaction.timestamp}"

async def check_vllm_health() -> Tuple[bool, Optional[str]]:
    """Check if vLLM server is healthy and model is loaded
    
    Returns:
        Tuple of (is_healthy, model_name)
    """
    if not vllm_client:
        return False, None
    try:
        models = await vllm_client.models.list()
        # Check if model is loaded
        if models.data and len(models.data) > 0:
            model_name = models.data[0].id
            return True, model_name
        return False, None
    except Exception as e:
        logger.warning(f"vLLM health check failed: {e}")
        return False, None

async def get_cached_prediction(transaction_id: str) -> Optional[Dict]:
    """Get cached prediction if available"""
    if not redis_client:
        return None
    try:
        cached = await redis_client.get(f"prediction:{transaction_id}")
        if cached:
            import json
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    return None

async def cache_prediction(transaction_id: str, prediction: Dict, ttl: int = None):
    """Cache prediction result"""
    if not redis_client:
        return
    try:
        import json
        await redis_client.setex(
            f"prediction:{transaction_id}",
            ttl or settings.cache_ttl,
            json.dumps(prediction)
        )
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

async def call_vllm_for_prediction(transaction: Transaction, request_id: str) -> Tuple[float, str]:
    """Call vLLM server for fraud prediction with proper error handling"""
    if not vllm_client:
        logger.info(f"[{request_id}] vLLM not available, using fallback")
        risk_score = 0.85 if transaction.amount > 1000 else 0.1
        explanation = f"High transaction amount (${transaction.amount}) detected." if risk_score > 0.5 else "Transaction appears normal."
        return risk_score, explanation
    
    try:
        # Format prompt for Llama-3.1
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnalyze this transaction for fraud risk.\n\nInput: {format_transaction_for_llm(transaction)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Get model name from vLLM (defaults to fraudguard-8b-merged)
        model_name = "fraudguard-8b-merged"
        try:
            models = await vllm_client.models.list()
            if models.data and len(models.data) > 0:
                model_name = models.data[0].id
        except Exception:
            pass  # Use default model name
        
        response = await vllm_client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
            stop=["<|eot_id|>", "\n\n"]
        )
        
        output_text = response.choices[0].text.strip()
        
        # Parse risk score from output
        risk_score = 0.5  # Default
        if "Risk Score:" in output_text:
            try:
                risk_str = output_text.split("Risk Score:")[1].split(")")[0].strip()
                risk_score = float(risk_str)
                risk_score = max(0.0, min(1.0, risk_score))  # Clamp to [0, 1]
            except (ValueError, IndexError) as e:
                logger.warning(f"[{request_id}] Failed to parse risk score: {e}")
                # Fallback: check for HIGH/LOW keywords
                if "HIGH" in output_text.upper():
                    risk_score = 0.85
                elif "LOW" in output_text.upper():
                    risk_score = 0.15
        else:
            # Fallback: check for HIGH/LOW keywords
            if "HIGH" in output_text.upper():
                risk_score = 0.85
            elif "LOW" in output_text.upper():
                risk_score = 0.15
        
        explanation = output_text
        logger.info(f"[{request_id}] vLLM prediction successful: risk_score={risk_score:.2f}")
        return risk_score, explanation
        
    except httpx.TimeoutException as e:
        logger.error(f"[{request_id}] vLLM timeout: {e}")
        risk_score = 0.85 if transaction.amount > 1000 else 0.1
        explanation = "Model inference timeout. Using fallback prediction."
        return risk_score, explanation
    except httpx.RequestError as e:
        logger.error(f"[{request_id}] vLLM request error: {e}")
        risk_score = 0.85 if transaction.amount > 1000 else 0.1
        explanation = "Model inference error. Using fallback prediction."
        return risk_score, explanation
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected vLLM error: {e}", exc_info=True)
        risk_score = 0.85 if transaction.amount > 1000 else 0.1
        explanation = "Model inference error. Using fallback prediction."
        return risk_score, explanation

def compute_shap_explanation(transaction: Transaction) -> Dict[str, float]:
    """Compute SHAP values for transaction features"""
    # For now, use a simple feature importance model
    # In production, you'd use a trained model with SHAP
    features = {
        "amount": transaction.amount,
        "merchant_risk": abs(hash(transaction.merchant)) % 100 / 100.0,  # Mock merchant risk
        "amount_normalized": min(transaction.amount / 5000.0, 1.0)
    }
    
    # Simple SHAP-like values (in production, use actual SHAP explainer)
    shap_values = {
        "amount": features["amount"] / 1000.0,  # Normalized contribution
        "merchant": features["merchant_risk"] * 0.3,
        "amount_normalized": features["amount_normalized"] * 0.5
    }
    
    return shap_values

# API Endpoints
@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    vllm_healthy, model_name = await check_vllm_health()
    redis_healthy = redis_client is not None and await redis_client.ping() if redis_client else False
    
    return {
        "status": "healthy",
        "service": "FraudGuard v2",
        "model": model_name or "Llama-3.1-8B-Instruct (fraudguard-8b-merged)",
        "model_loaded": vllm_healthy,
        "vllm_server": "connected" if vllm_healthy else "disconnected",
        "redis": "connected" if redis_healthy else "disconnected",
        "vllm_url": settings.vllm_server_url
    }

@app.get("/health", tags=["Health"])
async def detailed_health():
    """Detailed health check with component status and model verification"""
    vllm_healthy, model_name = await check_vllm_health()
    redis_healthy = redis_client is not None and await redis_client.ping() if redis_client else False
    
    # Determine overall status
    if vllm_healthy:
        overall_status = "healthy"
    elif not vllm_client:
        overall_status = "degraded"  # Fallback mode
    else:
        overall_status = "unhealthy"  # vLLM should be available but isn't
    
    return {
        "status": overall_status,
        "model": {
            "name": model_name or "fraudguard-8b-merged",
            "loaded": vllm_healthy,
            "status": "loaded" if vllm_healthy else "not_loaded"
        },
        "components": {
            "vllm": {
                "status": "healthy" if vllm_healthy else "unhealthy",
                "url": settings.vllm_server_url,
                "model_loaded": vllm_healthy
            },
            "redis": {
                "status": "healthy" if redis_healthy else "unhealthy",
                "url": settings.redis_url
            }
        }
    }

@app.post("/predict", response_model=FraudPrediction, tags=["Predictions"])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def predict_fraud(
    request: Request,
    transaction: Transaction,
    _: bool = Depends(verify_api_key)
):
    """Predict fraud risk for a transaction"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(f"[{request_id}] Prediction request: transaction_id={transaction.transaction_id}, amount={transaction.amount}")
    
    try:
        # Check cache first
        cached = await get_cached_prediction(transaction.transaction_id)
        if cached:
            logger.info(f"[{request_id}] Cache hit for transaction {transaction.transaction_id}")
            cached["request_id"] = request_id
            cached["latency_ms"] = (time.time() - start_time) * 1000
            return FraudPrediction(**cached)
        
        # Call vLLM for prediction
        risk_score, explanation = await call_vllm_for_prediction(transaction, request_id)
        
        latency = (time.time() - start_time) * 1000
        
        prediction = FraudPrediction(
            transaction_id=transaction.transaction_id,
            is_fraud=risk_score > 0.5,
            risk_score=risk_score,
            explanation=explanation,
            latency_ms=latency,
            request_id=request_id
        )
        
        # Cache the result
        await cache_prediction(transaction.transaction_id, prediction.model_dump())
        
        logger.info(f"[{request_id}] Prediction completed: risk_score={risk_score:.2f}, latency={latency:.2f}ms")
        return prediction
        
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/explain", response_model=ExplanationResponse, tags=["Explanations"])
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def explain_prediction(
    request: Request,
    explanation_request: ExplanationRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate SHAP-based explanation with natural language"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(f"[{request_id}] Explanation request: transaction_id={explanation_request.transaction_id}")
    
    try:
        # Convert to Transaction for processing
        transaction = Transaction(
            transaction_id=explanation_request.transaction_id,
            user_id=explanation_request.user_id,
            amount=explanation_request.amount,
            merchant=explanation_request.merchant,
            timestamp=explanation_request.timestamp
        )
        
        # Get prediction
        risk_score, base_explanation = await call_vllm_for_prediction(transaction, request_id)
        
        # Compute SHAP values
        shap_values = compute_shap_explanation(transaction)
        
        # Generate natural language explanation using vLLM
        natural_language_explanation = base_explanation
        if vllm_client:
            try:
                shap_summary = ", ".join([f"{k}: {v:.3f}" for k, v in shap_values.items()])
                explain_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nExplain why this transaction has a fraud risk score of {risk_score:.2f} based on these feature contributions: {shap_summary}. Provide a clear, natural language explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                # Get model name from vLLM (defaults to fraudguard-8b-merged)
                model_name = "fraudguard-8b-merged"
                try:
                    models = await vllm_client.models.list()
                    if models.data and len(models.data) > 0:
                        model_name = models.data[0].id
                except Exception:
                    pass  # Use default model name
                
                response = await vllm_client.completions.create(
                    model=model_name,
                    prompt=explain_prompt,
                    max_tokens=150,
                    temperature=0.2,
                    stop=["<|eot_id|>", "\n\n"]
                )
                
                natural_language_explanation = response.choices[0].text.strip()
            except Exception as e:
                logger.warning(f"[{request_id}] Error generating natural language explanation: {e}")
                shap_summary = ", ".join([f"{k}: {v:.3f}" for k, v in shap_values.items()])
                natural_language_explanation = f"Risk score: {risk_score:.2f}. Feature contributions: {shap_summary}"
        
        # Calculate feature importance (absolute values normalized)
        total_importance = sum(abs(v) for v in shap_values.values())
        feature_importance = {
            k: abs(v) / total_importance if total_importance > 0 else 0.0
            for k, v in shap_values.items()
        }
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Explanation completed: latency={latency:.2f}ms")
        
        return ExplanationResponse(
            transaction_id=transaction.transaction_id,
            shap_values=shap_values,
            natural_language_explanation=natural_language_explanation,
            feature_importance=feature_importance,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Explanation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        ).model_dump()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None  # Use our custom logging
    )
