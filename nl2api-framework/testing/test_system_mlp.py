import json
import os
import torch
import torch.nn as nn
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_TOOL_CALL = "davidferex/TFM_prueba5"
MLP_PATH        = "mlp_guardian.pt"
TOOLS_SPEC_PATH = "tools_spec.json"
MAX_LENGTH      = 128

with open(TOOLS_SPEC_PATH) as f:
    TOOLS_SPEC = json.load(f)
TOOLS_BY_NAME = {t["name"]: t for t in TOOLS_SPEC}

# ── Cargar modelo de tool calling ──────────────────────────────────────────────
print("⏳ Cargando modelo de tool calling...")
config_tool_call    = PeftConfig.from_pretrained(MODEL_TOOL_CALL)
base_tool_call      = AutoModelForCausalLM.from_pretrained(
    config_tool_call.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer_tool_call = AutoTokenizer.from_pretrained(MODEL_TOOL_CALL)
base_tool_call.resize_token_embeddings(len(tokenizer_tool_call))
model_tool_call     = PeftModel.from_pretrained(base_tool_call, MODEL_TOOL_CALL)
model_tool_call     = model_tool_call.merge_and_unload()  # merge para hidden states limpios
model_tool_call.eval()
for param in model_tool_call.parameters():
    param.requires_grad = False
print("✅ Modelo de tool calling cargado\n")

# ── Cargar MLP ─────────────────────────────────────────────────────────────────
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
    def forward(self, x):
        return self.net(x)

hidden_size = model_tool_call.config.hidden_size
mlp = GuardianMLP(hidden_size).to("cuda").to(torch.float32)
mlp.load_state_dict(torch.load(MLP_PATH, map_location="cuda"))
mlp.eval()
print(f"✅ MLP cargado (hidden_size={hidden_size})\n")

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_user_query(full_prompt: str) -> str:
    """Extrae solo la query del usuario — última línea antes de <end_of_turn>."""
    try:
        return full_prompt.split("<end_of_turn>")[0].split("\n")[-1].strip()
    except Exception:
        return full_prompt


def build_tool_call_prompt(user_query: str) -> str:
    tools_json = json.dumps([
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": {p["name"]: {"type": p["type"]} for p in t["parameters"]},
                    "required": [p["name"] for p in t["parameters"] if p["required"]]
                }
            }
        }
        for t in TOOLS_SPEC
    ])
    return (
        f"<bos><start_of_turn>human\n"
        f"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. "
        f"You may call one or more functions to assist with the user query. "
        f"Don't make assumptions about what values to plug into functions. "
        f"Here are the available tools:<tools> {tools_json} </tools>"
        f"Use the following pydantic model json schema for each tool call you will make: "
        f"{{'title': 'FunctionCall', 'type': 'object', 'properties': {{'arguments': {{'title': 'Arguments', 'type': 'object'}}, 'name': {{'title': 'Name', 'type': 'string'}}}}, 'required': ['arguments', 'name']}} "
        f"For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
        f"<tool_call>\n{{tool_call}}\n</tool_call>\n\n"
        f"{user_query}<end_of_turn><eos>\n"
        f"<start_of_turn>model<tool_call>"
    )


def extract_features_mlp(user_query: str) -> torch.Tensor:
    """Tokeniza solo la query limpia y extrae mean-pooled hidden states."""
    inputs = tokenizer_tool_call(
        user_query, truncation=True, max_length=MAX_LENGTH,
        padding="max_length", return_tensors="pt"
    )
    input_ids      = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    with torch.no_grad():
        outputs     = model_tool_call(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        mask        = attention_mask.unsqueeze(-1).float()
        query_vec   = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return query_vec.to(torch.float32)


def classify_completeness(user_query: str) -> dict:
    """Clasifica la query como completa o incompleta usando el MLP."""
    features = extract_features_mlp(user_query)
    with torch.no_grad():
        logits = mlp(features)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()
        pred   = torch.argmax(logits, dim=1).item()
    return {
        "complete":         pred == 1,
        "prob_complete":    round(probs[1], 4),
        "prob_incomplete":  round(probs[0], 4),
    }


def generate_tool_call(prompt: str) -> str:
    inputs     = tokenizer_tool_call(prompt, return_tensors="pt", add_special_tokens=False)
    inputs     = {k: v.to("cuda") for k, v in inputs.items()}
    outputs    = model_tool_call.generate(
        **inputs, max_new_tokens=200, do_sample=True,
        top_p=0.95, temperature=0.01, repetition_penalty=1.0,
        eos_token_id=tokenizer_tool_call.eos_token_id
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer_tool_call.decode(new_tokens, skip_special_tokens=True).strip()


def extract_tool_call(raw: str) -> dict | None:
    try:
        return json.loads(raw.split("</tool_call>")[0].strip())
    except Exception:
        return None


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run(user_query: str) -> dict:
    print(f"\n{'='*55}\nQuery: {user_query}\n{'='*55}")

    # Paso 1 — MLP: ¿es completa la query?
    mlp_result = classify_completeness(user_query)
    print(f"\n[Paso 1 - MLP] complete={mlp_result['complete']} "
          f"(P_complete={mlp_result['prob_complete']:.2%}, "
          f"P_incomplete={mlp_result['prob_incomplete']:.2%})")

    if not mlp_result["complete"]:
        # Query incompleta — no llamamos al tool caller
        # No sabemos qué falta exactamente (limitación del MLP)
        return {
            "status":           "incomplete",
            "mlp_confidence":   mlp_result["prob_incomplete"],
            "missing":          [],   # el MLP no sabe qué falta, solo que algo falta
        }

    # Paso 2 — Tool calling (solo si el MLP dice completa)
    raw_step2 = generate_tool_call(build_tool_call_prompt(user_query))
    print(f"\n[Paso 2 - raw] {raw_step2}")

    tool_call = extract_tool_call(raw_step2)
    if not tool_call or "name" not in tool_call:
        return {"error": "No se pudo extraer el tool call", "raw": raw_step2}

    tool_name = tool_call["name"]
    arguments = tool_call.get("arguments", {})
    print(f"[Paso 2] Tool: {tool_name} | Args: {arguments}")

    if tool_name not in TOOLS_BY_NAME:
        return {"error": f"Tool desconocida: {tool_name}", "tool_call": tool_call}

    return {"status": "ready", "tool": tool_name, "arguments": arguments}


if __name__ == "__main__":
    queries = [
        "I have an imbalanced dataset in the column 'SEXO' and want to balance it by generating 300 samples.",
        "Generate 500 imbalanced samples targeting the fraud_flag column.",
        "Balance my churn column using SMOTE.",
        "Generate 1000 time series samples using timegan, sequence key is session_id and time column is timestamp.",
        "I need 300 sequential samples, sequence key is device_id.",
    ]
    for q in queries:
        result = run(q)
        print(f"\n→ Resultado: {json.dumps(result, indent=2, ensure_ascii=False)}\n")