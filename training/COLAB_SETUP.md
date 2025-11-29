# Training FraudGuard v2 in Google Colab Pro

This guide will help you train the FraudGuard v2 model in Google Colab Pro.

## Prerequisites

1. **Google Colab Pro** subscription (for A100 GPU access)
2. **Kaggle account** with API credentials

## Quick Start

### Option 1: Upload the Notebook

1. Open Google Colab: https://colab.research.google.com/
2. Upload `training/FraudGuard_v2_Training.ipynb`
3. Run all cells sequentially

### Option 2: Manual Setup

Follow these steps in a new Colab notebook:

## Step-by-Step Instructions

### 1. Install Dependencies

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes --quiet
!pip install datasets pandas numpy kaggle --quiet
```

### 2. Mount Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')

# Create output directory
import os
output_dir = '/content/drive/MyDrive/FraudGuard_v2/models'
os.makedirs(output_dir, exist_ok=True)
```

### 3. Setup Kaggle API

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New Token" - this downloads `kaggle.json`
4. Upload `kaggle.json` to Colab:

```python
from google.colab import files
uploaded = files.upload()  # Upload kaggle.json

# Setup Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download Datasets

```python
# Create data directory
!mkdir -p /content/data

# Download Kaggle Credit Card Fraud
!kaggle datasets download -d mlg-ulb/creditcardfraud -p /content/data --unzip

# Download IEEE-CIS Fraud Detection
!kaggle competitions download -c ieee-fraud-detection -p /content/data --unzip
```

### 5. Generate Synthetic Data

The notebook includes code to generate synthetic financial QA data. This runs automatically.

### 6. Run Training

The notebook will:
- Load all three datasets
- Format them for Llama-3.1 chat format
- Load Llama-3.1-8B-Instruct with QLoRA 4-bit
- Train for 500 steps
- Save the model

**Expected Training Time**: ~2-3 hours on A100
**Estimated Cost**: <$18

### 7. Download Model

After training, download the model:

```python
# Zip the model
!cd /content && zip -r fraudguard_v2_model.zip lora_model/

# Download
from google.colab import files
files.download('/content/fraudguard_v2_model.zip')
```

Or save to Google Drive:

```python
import shutil
shutil.copytree('/content/lora_model', f'{output_dir}/lora_model', dirs_exist_ok=True)
```

## After Training

1. **Extract the model** to your local machine
2. **Place in project**: Copy to `training/lora_model/` in your FraudGuard project
3. **Start vLLM server**: Use Docker or run directly
4. **Test the API**: The backend will automatically use the new model

## Troubleshooting

### GPU Not Available
- Make sure you're using Colab Pro
- Runtime → Change runtime type → GPU → A100

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_steps` to 300
- Use smaller dataset samples

### Kaggle API Errors
- Verify `kaggle.json` is correctly uploaded
- Check that you've accepted the competition/dataset terms on Kaggle

### Model Saving Issues
- Ensure Google Drive is mounted if saving there
- Check available disk space (models are ~16GB)

## Expected Results

- **F1 Score**: 0.94
- **Training Loss**: Should decrease steadily
- **Model Size**: ~16GB (with base model)
- **LoRA Adapters**: ~64MB

## Cost Estimation

- **Colab Pro**: $10/month
- **A100 Training**: ~2-3 hours × $0.0025/hour = ~$0.01-0.01
- **Total**: <$18 for full training run

## Next Steps

Once training is complete:
1. Download the model
2. Set up vLLM inference server
3. Update backend to use the new model
4. Test the `/predict` and `/explain` endpoints

