import torch
import json
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ===============================
# Configuración
# ===============================
MODEL_BASE_ID     = "google/gemma-2-2b-it"
ADAPTER_ID        = "davidferex/TFM_prueba5"
MLP_PATH          = "mlp_guardian.pt"
TEST_DATASET_PATH = "test_mlp_v2.jsonl"
BATCH_SIZE        = 64     # ahora cabe con queries cortas
MAX_LENGTH        = 128
OUTPUT_ERROR_FILE = "errores_validacion.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 1. Cargar Tokenizador y Modelo
# ===============================
print("⏳ Cargando Tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

print("⏳ Cargando Modelo Base + LoRA...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0} if device == "cuda" else None
)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model = model.merge_and_unload()   # merge para representaciones más limpias
model.eval()
for param in model.parameters():
    param.requires_grad = False

# ===============================
# 2. Definir MLP (misma arquitectura que en training)
# ===============================
class GuardianMLP(nn.Module):
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
mlp = GuardianMLP(hidden_size).to(device).to(torch.float32)
mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
mlp.eval()

# ===============================
# 3. Preparar Dataset
# ===============================

def extract_user_query(full_prompt: str) -> str:
    """Extrae solo la query del usuario — última línea antes de <end_of_turn>."""
    try:
        content = full_prompt.split("<end_of_turn>")[0]
        return content.split("\n")[-1].strip()
    except Exception:
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


print("⏳ Preparando Dataset de Test...")
raw_dataset = load_dataset("json", data_files=TEST_DATASET_PATH, split="train")

# Verificación rápida del extractor
print(f"\n🔍 Ejemplo de extracción:")
print(f"  Query extraída : {extract_user_query(raw_dataset[0]['text'])}")
print(f"  Label          : {raw_dataset[0]['label']}\n")

tokenized_dataset = raw_dataset.map(preprocess, batched=True, remove_columns=raw_dataset.column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_loader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# 4. Extracción de features (mean pooling — igual que en training)
# ===============================

def extract_features(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]           # (B, T, H)
        mask        = attention_mask.unsqueeze(-1).float() # (B, T, 1)
        query_vec   = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        del outputs
        torch.cuda.empty_cache()
    return query_vec.to(torch.float32)

# ===============================
# 5. Loop de Evaluación
# ===============================
all_preds  = []
all_labels = []
error_log  = []

print(f"🚀 Iniciando evaluación en {device}...")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        features = extract_features(input_ids, attention_mask)
        logits   = mlp(features)
        preds    = torch.argmax(logits, dim=1)

        curr_preds  = preds.cpu().numpy()
        curr_labels = labels.cpu().numpy()

        all_preds.extend(curr_preds)
        all_labels.extend(curr_labels)

        # Registrar errores con la query limpia para facilitar el análisis
        for j in range(len(curr_preds)):
            if curr_preds[j] != curr_labels[j]:
                global_idx    = batch_idx * BATCH_SIZE + j
                original_text = raw_dataset[global_idx]["text"]
                clean_query   = extract_user_query(original_text)

                error_log.append({
                    "id"                : global_idx,
                    "query"             : clean_query,       # solo la query, más legible
                    "query_full"        : original_text,     # prompt completo por si lo necesitas
                    "label_real"        : int(curr_labels[j]),
                    "prediccion_modelo" : int(curr_preds[j]),
                    "estado"            : "INCOMPLETO" if curr_labels[j] == 0 else "COMPLETO"
                })

# ===============================
# 6. Métricas y Guardado
# ===============================
accuracy = accuracy_score(all_labels, all_preds)
f1       = f1_score(all_labels, all_preds, average="weighted")

with open(OUTPUT_ERROR_FILE, "w", encoding="utf-8") as f:
    json.dump(error_log, f, ensure_ascii=False, indent=4)

print("\n" + "="*40)
print("📊 RESULTADOS FINALES")
print("="*40)
print(f"✅ Accuracy : {accuracy:.4%}")
print(f"✅ F1 Score : {f1:.4f}")
print(f"❌ Errores  : {len(error_log)}")
print(f"📂 Errores guardados en: {OUTPUT_ERROR_FILE}")
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Incompleta", "Completa"]))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("="*40)