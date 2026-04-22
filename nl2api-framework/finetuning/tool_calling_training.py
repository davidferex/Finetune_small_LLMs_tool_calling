from enum import Enum
from functools import partial
import pandas as pd
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import TrainerCallback
import wandb

# 1. CONFIGURACIÓN INICIAL Y LOGINS
load_dotenv("train_HF.env")

# Autenticación en Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Autenticación en WandB
wandb_key = os.environ.get("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)

seed = 42
set_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class WandbCustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            wandb.log({
                "gpu_mem_allocated": torch.cuda.memory_allocated() / 1e9,
                "gpu_mem_reserved": torch.cuda.memory_reserved() / 1e9,
            })

# 2. TOKENIZER Y TOKENS ESPECIALES
model_name = "google/gemma-2-2b-it"

class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    tool_call="<tool_call>"
    eotool_call="</tool_call>"
    pad_token = "<pad>"
    eos_token = "<eos>"
    @classmethod
    def list(cls):
        return [c.value for c in cls]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    pad_token=ChatmlSpecialTokens.pad_token.value,
    additional_special_tokens=ChatmlSpecialTokens.list()
)

# Template corregido para Gemma 2
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

# 3. CARGA DEL MODELO Y REDIMENSIONAMIENTO 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation='eager',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Redimensionamos el vocabulario antes de aplicar LoRA
model.resize_token_embeddings(len(tokenizer))

# 4. CONFIGURACIÓN LORA
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["gate_proj","q_proj","o_proj","k_proj","down_proj","up_proj","v_proj"],
    task_type=TaskType.CAUSAL_LM
)

# 5. DATASET Y WANDB INIT
dataset = load_dataset("json", data_files="train_lora.jsonl", split="train")

username = "davidferex"
output_dir = "TFM_prueba7"

wandb.init(
    project="gemma-2b-tool-calling",
    name="lora-run-1",
    config={
        "model": model_name,
        "lr": 1e-4,
        "epochs": 1,
        "lora_r": 16,
    }
)

# 6. ENTRENAMIENTO
training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy="no",
    logging_steps=5,
    learning_rate=1e-4,
    max_grad_norm=1.0,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="wandb",
    bf16=True,
    num_train_epochs=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=False, 
    max_length=2048
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[WandbCustomCallback()]
)

trainer.train()

# 7. GUARDAR Y SUBIR
trainer.save_model()
trainer.push_to_hub(f"{username}/{output_dir}")

tokenizer.eos_token = ChatmlSpecialTokens.eos_token.value
tokenizer.push_to_hub(f"{username}/{output_dir}")

print("Entrenamiento completado y modelo subido a Hugging Face.")