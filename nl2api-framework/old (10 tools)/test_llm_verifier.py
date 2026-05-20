import json
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Load model ───────────────────────────────────────────────

MODEL_VERIFIER = "./TFM_verifier"
INPUT_DATASET  = "./verifier_test.json"

config = PeftConfig.from_pretrained(MODEL_VERIFIER)

base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_VERIFIER)
model = PeftModel.from_pretrained(base, MODEL_VERIFIER)
model.eval()


# ── Generation ───────────────────────────────────────────────

def generate(prompt: str, max_new_tokens: int = 120):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()


def extract_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None


# ── Load dataset ─────────────────────────────────────────────

with open(INPUT_DATASET, "r") as f:
    data = json.load(f)


# ── GLOBAL METRICS (VALIDITY CLASSIFICATION) ─────────────────

tp = fp = tn = fn = 0

# ── ERROR DETECTION METRICS (ONLY INVALID CASES) ─────────────

err_tp = err_fp = err_fn = 0

results = []

for sample in tqdm(data):

    gt = json.loads(sample["response"])
    prompt = sample["prompt"]

    pred_raw = generate(prompt)
    pred = extract_json(pred_raw) or {}

    gt_valid = gt.get("valid", None)
    pred_valid = pred.get("valid", None)

    # =========================
    # TASK A: VALIDITY CLASSIFICATION
    # =========================
    if gt_valid == True:
        if pred_valid == True:
            tp += 1
        else:
            fn += 1
    else:
        if pred_valid == False:
            tn += 1
        else:
            fp += 1

    # =========================
    # TASK B: ERROR DETECTION
    # =========================
    if gt_valid is False and pred_valid is False:

        gt_inc = gt.get("incorrect", [])
        pred_inc = pred.get("incorrect", [])

        gt_set = set(gt_inc)
        pred_set = set(pred_inc)

        err_tp += len(gt_set & pred_set)
        err_fp += len(pred_set - gt_set)
        err_fn += len(gt_set - pred_set)

    results.append({
        "gt": gt,
        "pred": pred,
        "gt_valid": gt_valid,
        "pred_valid": pred_valid
    })


# ── TASK A METRICS ───────────────────────────────────────────

accuracy = (tp + tn) / len(data)

precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) else 0)

# ── TASK B METRICS ───────────────────────────────────────────

err_precision = err_tp / (err_tp + err_fp) if (err_tp + err_fp) else 0
err_recall    = err_tp / (err_tp + err_fn) if (err_tp + err_fn) else 0
err_f1        = (2 * err_precision * err_recall /
                 (err_precision + err_recall)
                 if (err_precision + err_recall) else 0)


# ── PRINT RESULTS ────────────────────────────────────────────

print("\n==================== VERIFIER RESULTS ====================")

print("\n[Task A] Validity Classification")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")

print("\n[Task B] Error Detection (invalid cases only)")
print(f"Precision: {err_precision:.4f}")
print(f"Recall   : {err_recall:.4f}")
print(f"F1       : {err_f1:.4f}")

output = {
    "task_a_validity_classification": {
        "total": len(data),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    },
    "task_b_error_detection": {
        "precision": err_precision,
        "recall": err_recall,
        "f1": err_f1,
        "tp": err_tp,
        "fp": err_fp,
        "fn": err_fn
    },
    "examples": results
}

with open("verifier_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to verifier_results.json")