import json
from collections import defaultdict

TOOLS_SPEC_PATH = "tools_spec.json"


# Tool spec

def load_tools_spec(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_defaults_map(specs: list) -> dict:
    """
    Returns:
        {
            tool_name: {
                param_name: default_value
            }
        }
    Only includes params that have a default value.
    """
    defaults = {}

    for tool in specs:
        tool_defaults = {}

        for param in tool["parameters"]:
            if "default" in param:
                tool_defaults[param["name"]] = param["default"]

        defaults[tool["name"]] = tool_defaults

    return defaults


RAW_SPEC = load_tools_spec(TOOLS_SPEC_PATH)
DEFAULTS = build_defaults_map(RAW_SPEC)


# Helpers

def resolve_effective_args(args: dict, tool_name: str) -> dict:
    """
    Fill missing optional params with defaults.
    """
    tool_defaults = DEFAULTS.get(tool_name, {})

    effective = dict(args)

    for param, default_val in tool_defaults.items():
        if param not in effective:
            effective[param] = default_val

    return effective


def args_exact_match(pred_args: dict, true_args: dict, tool_name: str) -> bool:
    """
    Exact argument match after applying defaults.
    """
    pred_eff = resolve_effective_args(pred_args, tool_name)
    true_eff = resolve_effective_args(true_args, tool_name)

    return pred_eff == true_eff


def args_partial_score(pred_args: dict, true_args: dict, tool_name: str) -> dict:
    """
    Parameter-level scoring:
      - correct
      - missing
      - wrong_value
      - hallucinated
    """
    pred_eff = resolve_effective_args(pred_args, tool_name)
    true_eff = resolve_effective_args(true_args, tool_name)

    correct = 0
    missing = 0
    hallucinated = 0
    wrong_value = 0

    per_param = {}

    # Expected params
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

    # Extra params
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


# Evaluation

def evaluate():

    INPUT_DATASET = "results_LLAMA_10tools.json"
    OUTPUT_FILE = "results_LLAMA_numbers_10tools.json"

    with open(INPUT_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []

    # Exact metrics
    tool_correct = 0
    args_exact = 0
    both_correct = 0

    # Global partial metrics
    global_param_correct = 0
    global_param_missing = 0
    global_param_wrong = 0
    global_param_hallucinated = 0
    global_param_total = 0

    # Per tool
    per_tool = defaultdict(
        lambda: {
            "total": 0,
            "tool_correct": 0,
            "args_exact": 0,
            "param_correct": 0,
            "param_missing": 0,
            "param_wrong_value": 0,
            "param_hallucinated": 0,
            "param_total_expected": 0,
            "per_param": defaultdict(
                lambda: {
                    "total": 0,
                    "correct": 0,
                    "missing": 0,
                    "wrong_value": 0,
                    "hallucinated": 0,
                }
            ),
        }
    )

    for sample in dataset:

        true_tool = sample["tool"]
        true_args = sample["arguments"]

        pred_tool = sample["prediction"]["tool"]
        pred_args = sample["prediction"]["arguments"]

        # Tool selection
        tool_ok = pred_tool == true_tool

        # Exact args only if tool correct
        args_ok = (
            args_exact_match(pred_args, true_args, true_tool)
            if tool_ok else False
        )

        # Partial scoring always using true tool schema
        partial = args_partial_score(pred_args, true_args, true_tool)

        # Global exact counters
        if tool_ok:
            tool_correct += 1

        if args_ok:
            args_exact += 1

        if tool_ok and args_ok:
            both_correct += 1

        # Global partial counters
        global_param_correct += partial["correct"]
        global_param_missing += partial["missing"]
        global_param_wrong += partial["wrong_value"]
        global_param_hallucinated += partial["hallucinated"]
        global_param_total += partial["total_expected"]

        # Per-tool stats
        t = per_tool[true_tool]

        t["total"] += 1
        t["tool_correct"] += int(tool_ok)
        t["args_exact"] += int(args_ok)

        t["param_correct"] += partial["correct"]
        t["param_missing"] += partial["missing"]
        t["param_wrong_value"] += partial["wrong_value"]
        t["param_hallucinated"] += partial["hallucinated"]
        t["param_total_expected"] += partial["total_expected"]

        # Per parameter stats
        for param_name, outcome in partial["per_param"].items():
            pp = t["per_param"][param_name]
            pp["total"] += 1
            pp[outcome] += 1

        # Store example
        results.append({
            "true_tool": true_tool,
            "true_args": true_args,
            "pred_tool": pred_tool,
            "pred_args": pred_args,
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "partial": partial,
        })

    # Print global results

    n = len(dataset)
    gp = max(global_param_total, 1)

    print("\n" + "=" * 60)
    print(f"GLOBAL RESULTS ({n} examples)")
    print("=" * 60)

    print(f"Tool selection accuracy : {tool_correct/n:.2%} ({tool_correct}/{n})")
    print(f"Arg exact match         : {args_exact/n:.2%} ({args_exact}/{n})")
    print(f"Tool + Args correct     : {both_correct/n:.2%} ({both_correct}/{n})")

    print("-" * 60)

    print(f"Parameter correctness   : {global_param_correct/gp:.2%}")
    print(f"Parameter missing rate  : {global_param_missing/gp:.2%}")
    print(f"Wrong value rate        : {global_param_wrong/gp:.2%}")
    print(f"Hallucination rate      : {global_param_hallucinated/gp:.2%}")

    # Per tool

    print("\n" + "=" * 60)
    print("RESULTS PER TOOL")
    print("=" * 60)

    for tool_name, t in sorted(per_tool.items()):

        total = t["total"]
        pt = max(t["param_total_expected"], 1)

        print(f"\n▶ {tool_name} ({total} examples)")
        print(f"  Tool ok       : {t['tool_correct']/total:.2%}")
        print(f"  Args exact    : {t['args_exact']/total:.2%}")
        print(f"  Param correct : {t['param_correct']/pt:.2%}")
        print(f"  Missing       : {t['param_missing']/pt:.2%}")
        print(f"  Wrong value   : {t['param_wrong_value']/pt:.2%}")
        print(f"  Hallucinated  : {t['param_hallucinated']/pt:.2%}")

        print("  " + "─" * 42)
        print("  Per parameter:")

        for param_name, pp in sorted(t["per_param"].items()):

            ptotal = max(pp["total"], 1)

            print(
                f"    {param_name:<25}"
                f" correct={pp['correct']/ptotal:.0%}"
                f" missing={pp['missing']/ptotal:.0%}"
                f" wrong={pp['wrong_value']/ptotal:.0%}"
                f" halluc={pp['hallucinated']/ptotal:.0%}"
                f" (n={pp['total']})"
            )

    # Serialize JSON

    per_tool_serializable = {}

    for tool_name, t in per_tool.items():
        per_tool_serializable[tool_name] = {
            "total": t["total"],
            "tool_correct": t["tool_correct"],
            "args_exact": t["args_exact"],
            "param_correct": t["param_correct"],
            "param_missing": t["param_missing"],
            "param_wrong_value": t["param_wrong_value"],
            "param_hallucinated": t["param_hallucinated"],
            "param_total_expected": t["param_total_expected"],
            "per_param": dict(t["per_param"]),
        }

    output = {
        "global": {
            "total": n,
            "tool_accuracy": round(tool_correct / n, 4),
            "args_exact_match": round(args_exact / n, 4),
            "tool_and_args_correct": round(both_correct / n, 4),

            "param_correctness": round(global_param_correct / gp, 4),
            "param_missing_rate": round(global_param_missing / gp, 4),
            "param_wrong_value_rate": round(global_param_wrong / gp, 4),
            "param_hallucination_rate": round(global_param_hallucinated / gp, 4),
        },

        "per_tool": per_tool_serializable,
        "examples": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()