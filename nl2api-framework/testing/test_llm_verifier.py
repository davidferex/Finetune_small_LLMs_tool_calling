import json
import os
import torch
from tqdm import tqdm
from collections import defaultdict
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_VERIFIER = "./TFM_verifier"   # local o HF hub
INPUT_DATASET  = "./dataset_test/full_training_dataset_verifier.jsonl"
OUTPUT_FILE    = "results_verifier.json"

# ── Cargar modelo ──────────────────────────────────────────────────────────────
print("⏳ Cargando verifier...")
config    = PeftConfig.from_pretrained(MODEL_VERIFIER)
base      = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_VERIFIER)
base.resize_token_embeddings(len(tokenizer))
model     = PeftModel.from_pretrained(base, MODEL_VERIFIER)
model.eval()
print("✅ Listo\n")

# ── Helpers ────────────────────────────────────────────────────────────────────

def generate(prompt: str, max_new_tokens: int = 100) -> str:
    inputs     = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs     = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_json(raw: str) -> dict | None:
    # Tomar solo la primera línea que contenga un JSON completo
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    # Fallback: método original
    try:
        start = raw.find("{")
        end   = raw.find("}") + 1  # rfind → find, primer cierre
        return json.loads(raw[start:end])
    except Exception:
        return None


def build_prompt_from_messages(messages: list) -> str:
    """Reconstruye el prompt exactamente como en el training (chat template de Gemma)."""
    prompt = "<bos>"
    for msg in messages:
        prompt += f"<start_of_turn>{msg['role']}\n{msg['content'].strip()}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt


def missing_scores(pred_missing: list, true_missing: list) -> dict:
    pred_set = set(pred_missing)
    true_set = set(true_missing)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}

# ── Eval ───────────────────────────────────────────────────────────────────────

def evaluate():
    # Leer dataset — soporta .jsonl y .json (lista)
    samples = []
    with open(INPUT_DATASET, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    results           = []
    valid_correct     = 0
    missing_exact     = 0   # solo cuando true_valid=false
    n_incomplete      = 0   # ejemplos con true_valid=false

    agg_missing = {"tp": 0, "fp": 0, "fn": 0}  # para macro precision/recall

    for sample in tqdm(samples):
        messages  = sample["messages"]
        # El ground truth es el último mensaje (role=assistant)
        gt_raw    = messages[-1]["content"]
        gt        = extract_json(gt_raw) or {}

        # Prompt = todos los mensajes menos el último (el assistant de GT)
        prompt    = build_prompt_from_messages(messages[:-1])

        # Generar
        raw_pred  = generate(prompt)
        pred      = extract_json(raw_pred) or {}

        # ── Métricas ───────────────────────────────────────────────────────────
        # 1. valid/complete accuracy
        # El training usa "complete", el output del modelo también debería
        # Soportamos ambas keys por si hay inconsistencias
        def get_valid(d):
            if "valid" in d:    return bool(d["valid"])
            if "complete" in d: return bool(d["complete"])
            return None

        true_valid = get_valid(gt)
        pred_valid = get_valid(pred)
        valid_ok   = (pred_valid == true_valid)
        if valid_ok:
            valid_correct += 1

        # 2. missing params (solo cuando GT dice incomplete)
        true_missing = gt.get("missing", [])
        pred_missing = pred.get("missing", [])

        ms = None
        if true_valid is False:
            n_incomplete += 1
            exact = set(pred_missing) == set(true_missing)
            if exact:
                missing_exact += 1
            ms = missing_scores(pred_missing, true_missing)
            agg_missing["tp"] += ms["tp"]
            agg_missing["fp"] += ms["fp"]
            agg_missing["fn"] += ms["fn"]

        results.append({
            "prompt_user":  messages[0]["content"][:120] + "...",  # recorte para legibilidad
            "gt_raw":       gt_raw,
            "pred_raw":     raw_pred,
            "gt":           gt,
            "pred":         pred,
            "valid_ok":     valid_ok,
            "true_valid":   true_valid,
            "pred_valid":   pred_valid,
            "missing_scores": ms,
        })

    # ── Resumen ────────────────────────────────────────────────────────────────
    n = len(samples)
    tp, fp, fn = agg_missing["tp"], agg_missing["fp"], agg_missing["fn"]
    macro_p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    macro_r  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)
                if (macro_p + macro_r) > 0 else 0.0)

    print(f"\n{'='*55}")
    print(f"RESULTADOS VERIFIER ({n} ejemplos)")
    print(f"{'='*55}")
    print(f"  Valid/complete accuracy : {valid_correct/n:.2%}  ({valid_correct}/{n})")
    if n_incomplete:
        print(f"\n  — Sobre los {n_incomplete} ejemplos con GT=incomplete —")
        print(f"  Missing exact match    : {missing_exact/n_incomplete:.2%}  ({missing_exact}/{n_incomplete})")
        print(f"  Missing precision      : {macro_p:.2%}")
        print(f"  Missing recall         : {macro_r:.2%}")
        print(f"  Missing F1             : {macro_f1:.2%}")

    # ── Guardar ────────────────────────────────────────────────────────────────
    output = {
        "global": {
            "total":               n,
            "valid_accuracy":      round(valid_correct / n, 4),
            "n_incomplete_gt":     n_incomplete,
            "missing_exact_match": round(missing_exact / n_incomplete, 4) if n_incomplete else None,
            "missing_precision":   round(macro_p, 4),
            "missing_recall":      round(macro_r, 4),
            "missing_f1":          round(macro_f1, 4),
        },
        "examples": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Resultados guardados en {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()