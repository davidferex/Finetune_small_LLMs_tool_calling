import os
import json
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from enum import Enum
from transformers import AutoTokenizer

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. CONFIGURACIÓN INICIAL Y LOGINS
load_dotenv("train_HF.env")

# Autenticación en Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

TOOLS_SPEC_PATH = "tools_spec_full.json"

# ── Cargar modelo ──────────────────────────────────────────────────────────────
peft_model_id = "davidferex/TFM_all_TOOLS"

config    = PeftConfig.from_pretrained(peft_model_id)
model     = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto"
)


tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
    pad_token=ChatmlSpecialTokens.pad_token.value,
    additional_special_tokens=ChatmlSpecialTokens.list()
)

tokenizer.eos_token = ChatmlSpecialTokens.eos_token.value
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, peft_model_id)
model.to(torch.bfloat16)
model.eval()

# ── Tool spec ──────────────────────────────────────────────────────────────────

def load_tools_spec(path: str) -> list:
    with open(path, 'r') as f:
        return json.load(f)

def tools_to_json_schema(specs: list) -> list:
    schema_tools = []
    for tool in specs:
        properties = {}
        required = []

        for param in tool["parameters"]:
            p_name = param["name"]
            prop = {
                "type": param["type"],
                "description": param.get("description", "")
            }
            if "default" in param:
                prop["default"] = param["default"]

            properties[p_name] = prop

            if param.get("required", False):
                required.append(p_name)

        schema_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    return schema_tools


def build_defaults_map(specs: list) -> dict:
    defaults = {}
    for tool in specs:
        tool_defaults = {}
        for param in tool["parameters"]:
            if "default" in param:
                tool_defaults[param["name"]] = param["default"]
        defaults[tool["name"]] = tool_defaults
    return defaults


def build_base_prompt(tools_spec_path: str) -> str:
    raw_spec   = load_tools_spec(tools_spec_path)
    tools_list = tools_to_json_schema(raw_spec)

    return (
        f"<bos><start_of_turn>human"
        f"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags."
        f"You have to call one function to assist with the user query. Don't make assumptions about what values to plug into functions"
        f"Here are the available tools:<tools> {json.dumps(tools_list)} </tools>"
        f"For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
        f"<tool_call>\n{{tool_call}}\n</tool_call>"
    )

RAW_SPEC    = load_tools_spec(TOOLS_SPEC_PATH)
BASE_PROMPT = build_base_prompt(TOOLS_SPEC_PATH)
DEFAULTS    = build_defaults_map(RAW_SPEC)

# ── Memory helpers ─────────────────────────────────────────────────────────────

def get_gpu_memory_mb() -> float:
    """Returns current GPU memory allocated in MB (across all visible devices)."""
    if not torch.cuda.is_available():
        return 0.0
    total = sum(
        torch.cuda.memory_allocated(i)
        for i in range(torch.cuda.device_count())
    )
    return total / (1024 ** 2)


def get_gpu_reserved_mb() -> float:
    """Returns GPU memory reserved (cached) by PyTorch in MB."""
    if not torch.cuda.is_available():
        return 0.0
    total = sum(
        torch.cuda.memory_reserved(i)
        for i in range(torch.cuda.device_count())
    )
    return total / (1024 ** 2)


def reset_peak_memory_stats() -> None:
    """Resets PyTorch peak memory counters on all devices."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def get_peak_memory_mb() -> float:
    """Returns peak GPU memory allocated since the last reset, in MB."""
    if not torch.cuda.is_available():
        return 0.0
    total = sum(
        torch.cuda.max_memory_allocated(i)
        for i in range(torch.cuda.device_count())
    )
    return total / (1024 ** 2)

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


def resolve_effective_args(args: dict, tool_name: str) -> dict:
    tool_defaults = DEFAULTS.get(tool_name, {})
    effective = dict(args)
    for param, default_val in tool_defaults.items():
        if param not in effective:
            effective[param] = default_val
    return effective


def args_exact_match(pred_args: dict, true_args: dict, tool_name: str) -> bool:
    pred_eff = resolve_effective_args(pred_args, tool_name)
    true_eff = resolve_effective_args(true_args, tool_name)
    return pred_eff == true_eff


def args_partial_score(pred_args: dict, true_args: dict, tool_name: str) -> dict:
    pred_eff = resolve_effective_args(pred_args, tool_name)
    true_eff = resolve_effective_args(true_args, tool_name)

    correct      = 0
    missing      = 0
    hallucinated = 0
    wrong_value  = 0
    per_param    = {}

    for k, v in true_eff.items():
        if k not in pred_eff:
            missing += 1
            per_param[k] = "missing"
        elif pred_eff[k] == v:
            correct += 1
            per_param[k] = "correct"
        else:
            wrong_value += 1
            per_param[k] = "wrong_value"

    for k in pred_eff:
        if k not in true_eff:
            hallucinated += 1
            per_param[k] = "hallucinated"

    return {
        "correct": correct,
        "missing": missing,
        "wrong_value": wrong_value,
        "hallucinated": hallucinated,
        "total_expected": len(true_eff),
        "per_param": per_param,
    }

# ── Evaluación ─────────────────────────────────────────────────────────────────

def evaluate():
    INPUT_DATASET = "./dataset_test_all_tools/full_testing_dataset.json"
    OUTPUT_FILE   = "results_tool_caller_all_tools.json"

    with open(INPUT_DATASET, "r") as f:
        dataset = json.load(f)

    results      = []
    tool_correct = 0
    args_exact   = 0
    both_correct = 0

    global_param_correct      = 0
    global_param_missing      = 0
    global_param_wrong_value  = 0
    global_param_hallucinated = 0
    global_param_total        = 0

    # ── Memory accumulators ────────────────────────────────────────────────────
    # Measure baseline memory once before the loop (model weights + KV cache overhead)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    baseline_mem_mb = get_gpu_memory_mb()

    per_sample_peak_mb: list[float] = []   # peak during generate() per sample
    per_sample_delta_mb: list[float] = []  # peak - baseline per sample

    # Global per-tool stats
    per_tool = defaultdict(lambda: {
        "total": 0, "tool_correct": 0, "args_exact": 0,
        "param_correct": 0, "param_missing": 0,
        "param_wrong_value": 0, "param_hallucinated": 0,
        "param_total_expected": 0,
        "peak_mem_mb_sum": 0.0,          # sum of per-sample peaks for this tool
        "per_param": defaultdict(lambda: {
            "total": 0, "correct": 0, "missing": 0,
            "wrong_value": 0, "hallucinated": 0
        })
    })

    for sample in tqdm(dataset):
        user_query = sample["user_query"]
        true_tool  = sample["tool_call"]
        true_args  = sample["arguments"]

        prompt = (
            BASE_PROMPT
            + f"{user_query}<end_of_turn><eos>\n"
            + "<start_of_turn>model<tool_call>"
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Reset per-sample peak counter right before generation
        reset_peak_memory_stats()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id
            )

        # Capture peak memory for this sample immediately after generate()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sample_peak_mb  = get_peak_memory_mb()
        sample_delta_mb = max(0.0, sample_peak_mb - baseline_mem_mb)

        per_sample_peak_mb.append(sample_peak_mb)
        per_sample_delta_mb.append(sample_delta_mb)

        decoded        = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(decoded)
        generated_only = get_generated_only(decoded, prompt)

        pred_tool, pred_args = extract_tool_call(generated_only)

        tool_ok = pred_tool == true_tool
        args_ok = args_exact_match(pred_args, true_args, true_tool) if tool_ok else False
        partial = args_partial_score(pred_args, true_args, true_tool)

        global_param_correct      += partial["correct"]
        global_param_missing      += partial["missing"]
        global_param_wrong_value  += partial["wrong_value"]
        global_param_hallucinated += partial["hallucinated"]
        global_param_total        += partial["total_expected"]

        if tool_ok:
            tool_correct += 1
        if args_ok:
            args_exact += 1
        if tool_ok and args_ok:
            both_correct += 1

        # ── Acumular stats globales por tool ───────────────────────────────────
        t = per_tool[true_tool]
        t["total"]                += 1
        t["tool_correct"]         += int(tool_ok)
        t["args_exact"]           += int(args_ok)
        t["param_correct"]        += partial["correct"]
        t["param_missing"]        += partial["missing"]
        t["param_wrong_value"]    += partial["wrong_value"]
        t["param_hallucinated"]   += partial["hallucinated"]
        t["param_total_expected"] += partial["total_expected"]
        t["peak_mem_mb_sum"]      += sample_peak_mb

        # ── Acumular stats por parámetro individual ────────────────────────────
        for param_name, outcome in partial["per_param"].items():
            pp = t["per_param"][param_name]
            pp["total"] += 1
            pp[outcome] += 1

        results.append({
            "query":          user_query,
            "true_tool":      true_tool,
            "true_args":      true_args,
            "pred_tool":      pred_tool,
            "pred_args":      pred_args,
            "tool_ok":        tool_ok,
            "args_ok":        args_ok,
            "partial":        partial,
            "mem_peak_mb":    round(sample_peak_mb, 2),
            "mem_delta_mb":   round(sample_delta_mb, 2),
        })

    # ── Memory summary stats ───────────────────────────────────────────────────
    n = len(dataset)
    gp = global_param_total if global_param_total > 0 else 1
    avg_peak  = sum(per_sample_peak_mb)  / n if n else 0.0
    max_peak  = max(per_sample_peak_mb)  if per_sample_peak_mb  else 0.0
    min_peak  = min(per_sample_peak_mb)  if per_sample_peak_mb  else 0.0
    avg_delta = sum(per_sample_delta_mb) / n if n else 0.0
    max_delta = max(per_sample_delta_mb) if per_sample_delta_mb else 0.0

    # Final reserved memory (total pool held by PyTorch, includes fragmentation)
    final_reserved_mb = get_gpu_reserved_mb()

    # ── Imprimir resultados ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"RESULTADOS GLOBALES ({n} ejemplos)")
    print(f"{'='*55}")
    print(f"  Tool selection accuracy : {tool_correct/n:.2%} ({tool_correct}/{n})")
    print(f"  Arg exact match         : {args_exact/n:.2%}  ({args_exact}/{n})")
    print(f"  Tool + Args correct     : {both_correct/n:.2%} ({both_correct}/{n})")
    print(f"  Param correctness      : {global_param_correct/gp:.2%}")
    print(f"  Param missing rate     : {global_param_missing/gp:.2%}")
    print(f"  Param wrong value      : {global_param_wrong_value/gp:.2%}")
    print(f"  Param hallucination    : {global_param_hallucinated/gp:.2%}")

    print(f"\n{'='*55}")
    print("MEMORIA GPU (durante generate())")
    print(f"{'='*55}")
    print(f"  Baseline (pre-loop)     : {baseline_mem_mb:>8.1f} MB")
    print(f"  Avg peak  per sample    : {avg_peak:>8.1f} MB")
    print(f"  Max peak  per sample    : {max_peak:>8.1f} MB")
    print(f"  Min peak  per sample    : {min_peak:>8.1f} MB")
    print(f"  Avg delta (peak-base)   : {avg_delta:>8.1f} MB")
    print(f"  Max delta (peak-base)   : {max_delta:>8.1f} MB")
    print(f"  Final reserved (cached) : {final_reserved_mb:>8.1f} MB")

    print(f"\n{'='*55}")
    print("RESULTADOS POR TOOL")
    print(f"{'='*55}")
    for tool_name, t in sorted(per_tool.items()):
        total      = t["total"]
        p_total    = t["param_total_expected"] or 1
        avg_mem_t  = t["peak_mem_mb_sum"] / total if total else 0.0
        print(f"\n {tool_name} ({total} ejemplos)")
        print(f"    Tool ok       : {t['tool_correct']/total:.2%}")
        print(f"    Args exact    : {t['args_exact']/total:.2%}")
        print(f"    Param correct : {t['param_correct']/p_total:.2%}")
        print(f"    Param missing : {t['param_missing']/p_total:.2%}")
        print(f"    Wrong value   : {t['param_wrong_value']/p_total:.2%}")
        print(f"    Hallucinated  : {t['param_hallucinated']/p_total:.2%}")
        print(f"    Avg peak mem  : {avg_mem_t:.1f} MB")

        print(f"    {'─'*40}")
        print(f"    Stats por parámetro:")
        for param_name, pp in sorted(t["per_param"].items()):
            pt = pp["total"] or 1
            print(
                f"      {param_name:<25}"
                f"  correct={pp['correct']/pt:.0%}"
                f"  missing={pp['missing']/pt:.0%}"
                f"  wrong={pp['wrong_value']/pt:.0%}"
                f"  halluc={pp['hallucinated']/pt:.0%}"
                f"  (n={pp['total']})"
            )

    # ── Serializar a JSON ──────────────────────────────────────────────────────
    per_tool_serializable = {}
    for tool_name, t in per_tool.items():
        per_tool_serializable[tool_name] = {
            "total":                t["total"],
            "tool_correct":         t["tool_correct"],
            "args_exact":           t["args_exact"],
            "param_correct":        t["param_correct"],
            "param_missing":        t["param_missing"],
            "param_wrong_value":    t["param_wrong_value"],
            "param_hallucinated":   t["param_hallucinated"],
            "param_total_expected": t["param_total_expected"],
            "avg_peak_mem_mb":      round(t["peak_mem_mb_sum"] / t["total"], 2) if t["total"] else 0.0,
            "per_param":            dict(t["per_param"]),
        }

    output = {
        "global": {
            "total":               n,
            "tool_accuracy":       round(tool_correct / n, 4),
            "args_exact_match":    round(args_exact / n, 4),
            "both_correct":        round(both_correct / n, 4),
            "param_correctness": round(global_param_correct / gp, 4),
            "param_missing_rate": round(global_param_missing / gp, 4),
            "param_wrong_value_rate": round(global_param_wrong_value / gp, 4),
            "param_hallucination_rate": round(global_param_hallucinated / gp, 4),
            "memory": {
                "baseline_mb":       round(baseline_mem_mb, 2),
                "avg_peak_mb":       round(avg_peak, 2),
                "max_peak_mb":       round(max_peak, 2),
                "min_peak_mb":       round(min_peak, 2),
                "avg_delta_mb":      round(avg_delta, 2),
                "max_delta_mb":      round(max_delta, 2),
                "final_reserved_mb": round(final_reserved_mb, 2),
            },
        },
        "per_tool": per_tool_serializable,
        "examples": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n Resultados guardados en {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()