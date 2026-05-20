from enum import Enum
import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType
from dotenv import load_dotenv
import wandb

# ==========================================================
# ENV
# ==========================================================

load_dotenv("train_HF.env")

seed = 42
set_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

hf_token = os.environ.get("HF_TOKEN")

# ==========================================================
# CONFIG
# ==========================================================

model_name = "google/gemma-2-2b-it"
username   = "davidferex"
output_dir = "TFM_verifier"


dataset = load_dataset(
    "json",
    data_files="./verifier_train.json",
    split="train"
)

# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================================
# PREPROCESS
# Convertimos prompt/response a messages
# ==========================================================

def format_example(example):
    return {
        "prompt": example["prompt"],
        "completion": example["response"]
    }

dataset = dataset.map(format_example)

# ==========================================================
# MODEL
# ==========================================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
    token=hf_token
)

# ==========================================================
# LoRA
# ==========================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type=TaskType.CAUSAL_LM
)

# ==========================================================
# TRAINING
# ==========================================================

training_arguments = SFTConfig(
    output_dir=output_dir,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    save_strategy="no",
    eval_strategy="no",

    logging_steps=5,

    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,

    bf16=True,

    num_train_epochs=5,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    packing=True,
    max_length=768,

    report_to="wandb",
    push_to_hub=False
)

# ==========================================================
# WANDB
# ==========================================================

wandb.init(
    project="guardian-verifier",
    name="gemma2-verifier-json-v2",
    config={
        "model": model_name,
        "dataset": "verifier_train.json",
        "epochs": 5,
        "lr": 2e-4,
        "batch_size": 4,
        "grad_acc": 4,
        "lora_r": 16,
        "max_length": 768
    }
)

# ==========================================================
# TRAINER
# ==========================================================

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config
)

trainer.train()

# ==========================================================
# SAVE
# ==========================================================

trainer.save_model()

trainer.push_to_hub(f"{username}/{output_dir}")
tokenizer.push_to_hub(f"{username}/{output_dir}", token=hf_token)