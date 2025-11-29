FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM
# Note: vLLM should be able to load Unsloth models directly
RUN pip3 install --no-cache-dir \
    vllm==0.6.0 \
    transformers \
    torch \
    peft

# Create models directory
RUN mkdir -p /app/models

# Expose vLLM API port
EXPOSE 8000

# Copy startup script
COPY start_vllm.py /app/start_vllm.py
RUN chmod +x /app/start_vllm.py

# Run vLLM server
# Model path should be mounted at /app/models/fraudguard-8b-merged
CMD ["python3", "/app/start_vllm.py", "--model-path", "/app/models/fraudguard-8b-merged", "--host", "0.0.0.0", "--port", "8000"]

