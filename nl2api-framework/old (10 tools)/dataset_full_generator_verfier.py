import json
import random

# ==========================================================
# CONFIG
# ==========================================================

INPUT_DATASET = "verifier_dataset_train.json"
TOOLS_FILE    = "tools_spec.json"
OUTPUT_FILE   = "verifier_train.json"

SEED = 42
random.seed(SEED)


# ==========================================================
# LOAD DATA
# ==========================================================

with open(INPUT_DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

with open(TOOLS_FILE, "r", encoding="utf-8") as f:
    tools = json.load(f)


# ==========================================================
# INDEX TOOLS
# ==========================================================

tool_map = {tool["name"]: tool for tool in tools}


# ==========================================================
# HELPERS
# ==========================================================

def parse_parameters(tool):
    """
    Convierte el formato real:
    "parameters": [ {...}, {...} ]

    en required / optional
    """
    required = []
    optional = []

    for p in tool.get("parameters", []):
        if p.get("required", False):
            required.append(p["name"])
        else:
            optional.append(p["name"])

    return required, optional


def format_list(lst):
    return ", ".join(lst) if lst else "None"


def build_prompt(example):

    tool = tool_map[example["tool_call"]]

    required, optional = parse_parameters(tool)

    user_query = example["user_query"]
    params_json = json.dumps(
        example["arguments"],
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""You are a strict validator of tool calls.

Your task is to verify whether the tool call parameters are fully grounded in the user query.

TOOL NAME:
{tool["name"]}

TOOL DESCRIPTION:
{tool.get("description", "")}

REQUIRED PARAMETERS:
{format_list(required)}

OPTIONAL PARAMETERS:
{format_list(optional)}

USER QUERY:
"{user_query}"

TOOL CALL PARAMETERS (JSON):
{params_json}

Validation Rules:

1. Every REQUIRED parameter must be explicitly stated or clearly inferable from the query.
2. Every provided parameter value must be supported by the query.
3. Never assume information not present in the query.
4. If a parameter value is invented, contradictory, or unsupported, mark it incorrect.
5. Optional parameters may be omitted.

Return STRICT JSON ONLY.

If valid:
{{"valid": true}}

If invalid:
{{
  "valid": false,
  "incorrect": ["param1", "param2"]
}}

Return ONLY JSON.
"""

    return prompt


def build_response(example):

    if example["label"] == 1:
        return json.dumps(
            {"valid": True},
            ensure_ascii=False
        )

    return json.dumps(
        {
            "valid": False,
            "incorrect": example["wrong_parameters"]
        },
        ensure_ascii=False
    )


# ==========================================================
# BUILD DATASET
# ==========================================================

final_data = []

for ex in dataset:

    final_data.append({
        "prompt": build_prompt(ex),
        "response": build_response(ex)
    })


# Mezclar orden
random.shuffle(final_data)


# ==========================================================
# SAVE
# ==========================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

print(f"Saved {len(final_data)} examples to {OUTPUT_FILE}")