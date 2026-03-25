import os
import json
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ── Cargar modelo ──────────────────────────────────────────────────────────────
peft_model_id = "./TFM_prueba5"

config    = PeftConfig.from_pretrained(peft_model_id)
model     = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, peft_model_id)
model.to(torch.bfloat16)
model.eval()

# ── Prompt base (FIX: required corregidos) ─────────────────────────────────────
BASE_PROMPT = """<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [
{"type": "function", "function": {"name": "FastPrototypePreset", "parameters": {"type": "object", "properties": {"n_samples": {"type": "integer"}}, "required": ["n_samples"]}}},
{"type": "function", "function": {"name": "HighFidelityPreset", "parameters": {"type": "object", "properties": {"n_samples": {"type": "integer"}, "auto_report": {"type": "boolean"}}, "required": ["n_samples"]}}},
{"type": "function", "function": {"name": "ImbalancedGeneratorPreset", "parameters": {"type": "object", "properties": {"n_samples": {"type": "integer"}, "target_col": {"type": "string"}, "imbalance_ratio": {"type": "number"}}, "required": ["n_samples", "target_col"]}}},
{"type": "function", "function": {"name": "TimeSeriesPreset", "parameters": {"type": "object", "properties": {"n_samples": {"type": "integer"}, "sequence_key": {"type": "string"}, "time_col": {"type": "string"}, "method": {"type": "string"}}, "required": ["n_samples", "time_col"]}}},
{"type": "function", "function": {"name": "BalancedDataGeneratorPreset", "parameters": {"type": "object", "properties": {"n_samples": {"type": "integer"}, "target_col": {"type": "string"}}, "required": ["n_samples", "target_col"]}}}
] </tools>
Use the following pydantic model json schema for each tool call you will make:
{'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_tool_call(text: str) -> tuple[str | None, dict]:
    text  = text.strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None, {}
    try:
        parsed = json.loads(text[start:end + 1])
        return parsed.get("name"), parsed.get("arguments", {})
    except json.JSONDecodeError:
        return None, {}


def get_generated_only(decoded: str, prompt: str) -> str:
    idx = decoded.find(prompt)
    if idx >= 0:
        return decoded[idx + len(prompt):].strip()
    return decoded.strip()


def args_exact_match(pred_args: dict, true_args: dict) -> bool:
    if set(pred_args.keys()) != set(true_args.keys()):
        return False
    for k, v in true_args.items():
        if pred_args.get(k) != v:
            return False
    return True


def args_partial_score(pred_args: dict, true_args: dict) -> dict:
    correct      = 0
    missing      = 0
    hallucinated = 0
    wrong_value  = 0

    for k, v in true_args.items():
        if k not in pred_args:
            missing += 1
        elif pred_args[k] == v:
            correct += 1
        else:
            wrong_value += 1

    for k in pred_args:
        if k not in true_args:
            hallucinated += 1

    return {
        "correct": correct,
        "missing": missing,
        "wrong_value": wrong_value,
        "hallucinated": hallucinated,
        "total_expected": len(true_args),
    }

# ── Evaluación ─────────────────────────────────────────────────────────────────

def evaluate():
    INPUT_DATASET = "test.json"
    OUTPUT_FILE   = "results_tool_caller.json"

    with open(INPUT_DATASET, "r") as f:
        dataset = json.load(f)

    results          = []
    tool_correct     = 0
    args_exact       = 0
    both_correct     = 0

    per_tool = defaultdict(lambda: {
        "total": 0, "tool_correct": 0, "args_exact": 0,
        "param_correct": 0, "param_missing": 0,
        "param_wrong_value": 0, "param_hallucinated": 0,
        "param_total_expected": 0
    })

    for sample in tqdm(dataset):

        # ── FIX: nuevo formato dataset ─────────────────────────────
        user_query = sample["query"]
        true_tool  = sample["tool_call"]["name"]
        true_args  = sample["tool_call"]["arguments"]

        prompt = (
            BASE_PROMPT
            + f"{user_query}<end_of_turn><eos>\n"
            + "<start_of_turn>model<tool_call>"
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded        = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_only = get_generated_only(decoded, prompt)

        pred_tool, pred_args = extract_tool_call(generated_only)

        # ── Métricas ───────────────────────────────────────────────
        tool_ok = pred_tool == true_tool
        args_ok = args_exact_match(pred_args, true_args) if tool_ok else False
        partial = args_partial_score(pred_args, true_args)

        if tool_ok:
            tool_correct += 1
        if args_ok:
            args_exact += 1
        if tool_ok and args_ok:
            both_correct += 1

        t = per_tool[true_tool]
        t["total"]               += 1
        t["tool_correct"]        += int(tool_ok)
        t["args_exact"]          += int(args_ok)
        t["param_correct"]       += partial["correct"]
        t["param_missing"]       += partial["missing"]
        t["param_wrong_value"]   += partial["wrong_value"]
        t["param_hallucinated"]  += partial["hallucinated"]
        t["param_total_expected"]+= partial["total_expected"]

        results.append({
            "query": user_query,
            "true_tool": true_tool,
            "true_args": true_args,
            "pred_tool": pred_tool,
            "pred_args": pred_args,
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "partial": partial,
        })

    # ── Resumen ───────────────────────────────────────────────────
    n = len(dataset)
    print(f"\n{'='*50}")
    print(f"RESULTADOS GLOBALES ({n} ejemplos)")
    print(f"{'='*50}")
    print(f"  Tool selection accuracy : {tool_correct/n:.2%} ({tool_correct}/{n})")
    print(f"  Arg exact match         : {args_exact/n:.2%}  ({args_exact}/{n})")
    print(f"  Tool + Args correct     : {both_correct/n:.2%} ({both_correct}/{n})")

    print(f"\n{'='*50}")
    print("RESULTADOS POR TOOL")
    print(f"{'='*50}")
    for tool_name, t in sorted(per_tool.items()):
        total   = t["total"]
        p_total = t["param_total_expected"] or 1
        print(f"\n  {tool_name} ({total} ejemplos)")
        print(f"    Tool ok       : {t['tool_correct']/total:.2%}")
        print(f"    Args exact    : {t['args_exact']/total:.2%}")
        print(f"    Param correct : {t['param_correct']/p_total:.2%}")
        print(f"    Param missing : {t['param_missing']/p_total:.2%}")
        print(f"    Wrong value   : {t['param_wrong_value']/p_total:.2%}")
        print(f"    Hallucinated  : {t['param_hallucinated']/p_total:.2%}")

    # ── Guardar ───────────────────────────────────────────────────
    output = {
        "global": {
            "total": n,
            "tool_accuracy": round(tool_correct / n, 4),
            "args_exact_match": round(args_exact / n, 4),
            "both_correct": round(both_correct / n, 4),
        },
        "per_tool": dict(per_tool),
        "examples": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Resultados guardados en {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()