#!/usr/bin/env python3
"""
Start vLLM server for FraudGuard v2
Loads the fine-tuned Llama-3.1-8B model from inference/model/fraudguard-8b-merged
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_model_files(model_path: str) -> bool:
    """Check if model directory contains required files"""
    required_files = ["config.json", "tokenizer.json"]
    model_path_obj = Path(model_path)
    
    if not model_path_obj.exists():
        return False
    
    # Check for at least config.json
    config_file = model_path_obj / "config.json"
    if not config_file.exists():
        return False
    
    # Check for model files (could be .safetensors, .bin, or .pt)
    has_model_file = any(
        list(model_path_obj.glob("*.safetensors")) or
        list(model_path_obj.glob("*.bin")) or
        list(model_path_obj.glob("*.pt"))
    )
    
    return has_model_file

def main():
    parser = argparse.ArgumentParser(description="Start vLLM server for FraudGuard")
    default_model_path = os.environ.get("MODEL_PATH", "/app/models/fraudguard-8b-merged")
    
    parser.add_argument("--model-path", type=str, default=default_model_path,
                       help="Path to the fine-tuned model (default: /app/models/fraudguard-8b-merged)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    
    args = parser.parse_args()
    
    # Resolve absolute path
    model_path = os.path.abspath(args.model_path)
    
    # Check if model exists and has required files
    print(f"Checking model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model directory not found at {model_path}")
        print(f"   Expected path: {model_path}")
        print(f"   Please ensure the model is located at: inference/model/fraudguard-8b-merged")
        sys.exit(1)
    
    if not check_model_files(model_path):
        print(f"❌ ERROR: Model directory exists but missing required files")
        print(f"   Expected files: config.json, and at least one model file (.safetensors, .bin, or .pt)")
        print(f"   Model path: {model_path}")
        sys.exit(1)
    
    print("✅ Model files found")
    print(f"🚀 Starting vLLM server for FraudGuard v2...")
    print(f"   Model: fraudguard-8b-merged (Llama-3.1-8B)")
    print(f"   Model path: {model_path}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   GPU Memory: {args.gpu_memory_utilization * 100}%")
    
    # Import vLLM
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
    except ImportError as e:
        print(f"❌ ERROR: vLLM not installed")
        print(f"   Install with: pip install vllm")
        print(f"   Error: {e}")
        sys.exit(1)
    
    # Start vLLM server
    try:
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", args.host,
            "--port", str(args.port),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--trust-remote-code"
        ]
        
        print(f"📡 Starting server with command:")
        print(f"   {' '.join(cmd)}")
        print("")
        
        # Run vLLM server (this will block)
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: vLLM server failed to start")
        print(f"   Exit code: {e.returncode}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERROR: Unexpected error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

