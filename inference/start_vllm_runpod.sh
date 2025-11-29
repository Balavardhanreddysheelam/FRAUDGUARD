#!/bin/bash
# vLLM Startup Script for RunPod
# Upload this to your RunPod instance and run it

echo "🚀 Starting vLLM Server for FraudGuard on RunPod"
echo "=================================================="

# Model path - adjust if your model is in a different location
MODEL_PATH="/workspace/fraudguard-8b-merged"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model not found at $MODEL_PATH"
    echo "   Please upload your model to RunPod first"
    echo "   Or update MODEL_PATH in this script"
    exit 1
fi

echo "✅ Model found at: $MODEL_PATH"
echo ""

# Install vLLM if not already installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "📦 Installing vLLM..."
    pip install vllm
fi

echo "🚀 Starting vLLM server..."
echo "   Model: fraudguard-8b-merged"
echo "   Host: 0.0.0.0"
echo "   Port: 8000"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9



