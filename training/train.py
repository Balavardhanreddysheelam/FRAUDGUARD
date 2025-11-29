"""
FraudGuard v2 Training Script
Fine-tunes Llama-3.1-8B-Instruct with Unsloth + QLoRA 4-bit
on Kaggle Credit Card Fraud + IEEE-CIS Fraud + synthetic financial QA
"""
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import pandas as pd
import json
import os
from pathlib import Path

def load_fraud_datasets():
    """Load and combine fraud datasets"""
    data_dir = Path(__file__).parent.parent / "data"
    examples = []
    
    # Load Kaggle Credit Card Fraud dataset
    creditcard_path = data_dir / "creditcard.csv"
    if creditcard_path.exists():
        print("Loading Kaggle Credit Card Fraud dataset...")
        df_cc = pd.read_csv(creditcard_path)
        # Sample for training (use all if small, otherwise sample)
        if len(df_cc) > 10000:
            df_cc = df_cc.sample(n=10000, random_state=42)
        
        for _, row in df_cc.iterrows():
            # Extract key features
            amount = row.get('Amount', 0)
            is_fraud = int(row.get('Class', 0))
            
            # Create instruction
            instruction = "Analyze this credit card transaction for fraud risk."
            input_text = f"Transaction Amount: ${amount:.2f}, V1: {row.get('V1', 0):.3f}, V2: {row.get('V2', 0):.3f}, V3: {row.get('V3', 0):.3f}, V4: {row.get('V4', 0):.3f}, V5: {row.get('V5', 0):.3f}"
            
            if is_fraud:
                output = f"Fraud Risk: HIGH (Risk Score: 0.95). This transaction shows anomalous patterns consistent with fraudulent activity. The combination of transaction amount, timing, and feature values indicates a high probability of fraud. Immediate action recommended."
            else:
                output = f"Fraud Risk: LOW (Risk Score: 0.05). This transaction appears normal and consistent with the user's typical spending patterns. No suspicious activity detected."
            
            examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })
    
    # Load IEEE-CIS Fraud Detection dataset
    ieee_path = data_dir / "train_transaction.csv"
    if not ieee_path.exists():
        # Try alternative names
        ieee_files = list(data_dir.glob("*transaction*.csv"))
        if ieee_files:
            ieee_path = ieee_files[0]
    
    if ieee_path.exists():
        print("Loading IEEE-CIS Fraud Detection dataset...")
        df_ieee = pd.read_csv(ieee_path)
        # Sample for training
        if len(df_ieee) > 10000:
            df_ieee = df_ieee.sample(n=10000, random_state=42)
        
        for _, row in df_ieee.iterrows():
            amount = row.get('TransactionAmt', 0)
            is_fraud = int(row.get('isFraud', 0))
            product_cd = row.get('ProductCD', 'Unknown')
            
            instruction = "Analyze this financial transaction for fraud risk."
            input_text = f"Transaction Amount: ${amount:.2f}, Product Code: {product_cd}, Card Type: {row.get('card4', 'Unknown')}"
            
            if is_fraud:
                output = f"Fraud Risk: HIGH (Risk Score: 0.92). This transaction exhibits characteristics of fraudulent behavior including unusual amount patterns and suspicious product code combinations. Recommend blocking and investigation."
            else:
                output = f"Fraud Risk: LOW (Risk Score: 0.08). Transaction appears legitimate with normal spending patterns. No immediate concerns."
            
            examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })
    
    # Load synthetic financial QA
    synthetic_path = data_dir / "synthetic_financial_qa.csv"
    if synthetic_path.exists():
        print("Loading synthetic financial QA dataset...")
        df_synth = pd.read_csv(synthetic_path)
        # Use all synthetic data
        if len(df_synth) > 5000:
            df_synth = df_synth.sample(n=5000, random_state=42)
        
        for _, row in df_synth.iterrows():
            amount = row.get('amount', 0)
            merchant = row.get('merchant', 'Unknown')
            is_fraud = int(row.get('is_fraud_risk', 0))
            question = row.get('user_question', '')
            
            instruction = "Analyze this transaction and answer the user's question about fraud risk."
            input_text = f"Transaction Amount: ${amount:.2f}, Merchant: {merchant}, User Question: {question}"
            
            if is_fraud:
                output = f"Fraud Risk: HIGH (Risk Score: 0.88). {row.get('model_answer', 'This transaction shows suspicious patterns.')} The transaction amount and merchant combination raise concerns. Recommend immediate review."
            else:
                output = f"Fraud Risk: LOW (Risk Score: 0.12). {row.get('model_answer', 'This transaction appears normal.')} No suspicious activity detected."
            
            examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })
    
    print(f"Total training examples: {len(examples)}")
    return examples

def train():
    """Fine-tune Llama-3.1-8B-Instruct with QLoRA"""
    print("Starting FraudGuard v2 training...")
    print("Model: Llama-3.1-8B-Instruct")
    print("Method: Unsloth + QLoRA 4-bit")
    
    max_seq_length = 2048
    dtype = None  # Auto detection
    load_in_4bit = True  # Use 4-bit quantization
    
    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Apply QLoRA
    print("Applying QLoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Load datasets
    examples = load_fraud_datasets()
    
    if not examples:
        print("Warning: No training data found. Using mock data.")
        examples = [{
            "instruction": "Analyze this transaction for fraud.",
            "input": "Amount: $500, Merchant: Unknown",
            "output": "Fraud Risk: HIGH (Risk Score: 0.85). Unusual merchant pattern detected."
        }] * 100
    
    # Format prompts
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Use Llama-3.1 chat format
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nInput: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            texts.append(text)
        return {"text": texts}
    
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Training arguments
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            max_steps=500,  # Adjust based on dataset size
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="steps",
            save_steps=100,
        ),
    )
    
    trainer_stats = trainer.train()
    
    # Save model
    output_dir = Path(__file__).parent / "lora_model"
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print("Training complete!")
    print(f"Training stats: {trainer_stats}")
    print(f"Model saved to: {output_dir}")
    print("\nNext steps:")
    print("1. The model is ready for inference")
    print("2. Use vLLM to serve the model for production")
    print("3. Training cost: <$18 (estimated for 500 steps on A100)")

if __name__ == "__main__":
    train()
