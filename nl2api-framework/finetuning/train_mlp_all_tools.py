import os
from enum import Enum
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, ClassLabel
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from dotenv import load_dotenv
from huggingface_hub import login

# ESTO TIENE QUE SER LA PRIMERA LÍNEA DEL SCRIPT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(f"Número de GPUs visibles: {torch.cuda.device_count()}")
print(f"GPU en uso: {torch.cuda.get_device_name(0)}")

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
    
# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
MODEL_BASE_ID   = "google/gemma-2-2b-it"
ADAPTER_ID      = "davidferex/TFM_all_tools"
DATASET_PATH    = "train_mlp_all_tools.jsonl"
OUTPUT_MLP_PATH = "mlp_guardian_all_tools.pt"

BATCH_SIZE   = 16    # Ahora cabe sin problema al tokenizar solo la query
EPOCHS       = 15
LEARNING_RATE = 1e-4
MAX_LENGTH   = 128   # Las queries limpias raramente superan 80 tokens
SEED         = 42

torch.manual_seed(SEED)

# ==========================================
# 2. CARGA DEL MODELO
# ==========================================

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
        MODEL_BASE_ID,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list()
    )


print("⏳ Cargando Modelo Base y Adaptador...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
base_model.resize_token_embeddings(len(tokenizer))

# Merge LoRA → modelo denso para representaciones más limpias
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model = model.merge_and_unload()
model.eval()

for param in model.parameters():
    param.requires_grad = False

print(f" Modelo cargado. Hidden size: {model.config.hidden_size}")

# ==========================================
# 3. ARQUITECTURA DEL MLP
# ==========================================
class GuardianMLP(nn.Module):
    """
    Clasificador binario sobre mean-pooled hidden states de Gemma.
    Entrada: vector de hidden_size (2304 para Gemma 2 2B)
    Salida:  logits [incompleta, completa]
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, 2)   # [0: Incompleta, 1: Completa]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


hidden_size = model.config.hidden_size
mlp = GuardianMLP(hidden_size).to("cuda").to(torch.float32)

# ==========================================
# 4. PREPARACIÓN DE DATOS
# ==========================================

def extract_user_query(full_prompt: str) -> str:
    # La query está justo antes de <end_of_turn>, después del último \n
    try:
        content = full_prompt.split("<end_of_turn>")[0]  # quita el cierre
        return content.split("\n")[-1].strip()           # última línea = query
    except:
        return full_prompt


def preprocess(examples):
    clean_queries = [extract_user_query(t) for t in examples["text"]]
    tokenized = tokenizer(
        clean_queries,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokenized["label"] = examples["label"]
    return tokenized


print(" Preparando Dataset...")
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")



raw_dataset = raw_dataset.cast_column(
    "label",
    ClassLabel(names=["Incompleta", "Completa"])
)

# Split train / val 90-10
split = raw_dataset.train_test_split(test_size=0.1, seed=SEED, stratify_by_column="label")
train_ds = split["train"]
val_ds   = split["test"]

train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f" Dataset listo — Train: {len(train_ds)} | Val: {len(val_ds)}")

# ==========================================
# 5. EXTRACCIÓN DE HIDDEN STATES (mean pooling)
# ==========================================

def extract_features(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling sobre el último hidden state, excluyendo tokens de padding.
    Mucho más robusto que last-token para detectar ausencia de valores concretos.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]           # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()       # (B, T, 1)
        sum_hidden = (last_hidden * mask).sum(dim=1)      # (B, H)
        query_vector = sum_hidden / mask.sum(dim=1)       # (B, H)  mean pool
    return query_vector.to(torch.float32)


# ==========================================
# 6. ENTRENAMIENTO
# ==========================================
sample = raw_dataset[0]
print(extract_user_query(sample["text"]))
print(sample["label"])

wandb.init(
    project="guardian-mlp",
    name="gemma-mlp-v1",
    config={
        "model_base": MODEL_BASE_ID,
        "adapter": ADAPTER_ID,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "hidden_size": hidden_size
    }
)

optimizer = torch.optim.AdamW(mlp.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0

print("\n Iniciando entrenamiento...\n")

for epoch in range(EPOCHS):

    # ── Train ──────────────────────────────────────────────────────────────
    mlp.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for batch in train_loader:
        input_ids     = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels        = batch["label"].to("cuda")

        features = extract_features(input_ids, attention_mask)

        optimizer.zero_grad()
        logits = mlp(features)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    scheduler.step()

    # ── Validation ─────────────────────────────────────────────────────────
    mlp.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels         = batch["label"].to("cuda")

            features = extract_features(input_ids, attention_mask)
            logits   = mlp(features)
            loss     = criterion(logits, labels)

            val_loss    += loss.item()
            preds        = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss / len(train_loader),
        "train_acc": train_acc,
        "val_loss": val_loss / len(val_loader),
        "val_acc": val_acc,
        "lr": scheduler.get_last_lr()[0]
    })

    if torch.cuda.is_available():
        wandb.log({
            "gpu_mem_allocated": torch.cuda.memory_allocated() / 1e9,
            "gpu_mem_reserved": torch.cuda.memory_reserved() / 1e9,
        })

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2%}")

    # Guardar el mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(mlp.state_dict(), OUTPUT_MLP_PATH)
        print(f"  Nuevo mejor modelo guardado (val_acc={val_acc:.2%})")

        wandb.log({
            "best_val_acc": best_val_acc,
            "best_epoch": epoch + 1
        })

# ==========================================
# 7. REPORTE FINAL
# ==========================================
print(f"\n Entrenamiento completado. Mejor val_acc: {best_val_acc:.2%}")
print(f"   Modelo guardado en: {OUTPUT_MLP_PATH}\n")

# Cargar el mejor modelo para el reporte final
mlp.load_state_dict(torch.load(OUTPUT_MLP_PATH))
mlp.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids      = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels         = batch["label"].to("cuda")

        features = extract_features(input_ids, attention_mask)
        logits   = mlp(features)
        preds    = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

report = classification_report(
    all_labels,
    all_preds,
    target_names=["Incompleta", "Completa"],
    output_dict=True
)
print(" Classification Report (val set):")
print(classification_report(all_labels, all_preds, target_names=["Incompleta", "Completa"]))
wandb.log({
    "final_accuracy": report["accuracy"],
    "precision_completa": report["Completa"]["precision"],
    "recall_completa": report["Completa"]["recall"],
    "f1_completa": report["Completa"]["f1-score"],
    "precision_incompleta": report["Incompleta"]["precision"],
    "recall_incompleta": report["Incompleta"]["recall"],
    "f1_incompleta": report["Incompleta"]["f1-score"],
})
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=["Incompleta", "Completa"]
    )
})