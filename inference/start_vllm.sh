#!/bin/bash
# Start vLLM server for FraudGuard v2
# This script loads the fine-tuned Llama-3.1-8B model and starts the vLLM server

MODEL_PATH=${MODEL_PATH:-/app/models/lora_model}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "Starting vLLM server..."
echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure the fine-tuned model is available at this path"
    exit 1
fi

# Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code

