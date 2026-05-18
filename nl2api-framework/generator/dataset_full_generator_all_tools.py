import json
import os
import random

# ── Cargar spec desde fichero ──────────────────────────────────────────────────
with open("tools_spec_full.json", "r") as f:
    TOOLS_SPEC = json.load(f)

# Mapeo de tipos del spec al formato JSON Schema del prompt
TYPE_MAP = {
    "int":    "integer",
    "float":  "number",
    "bool":   "boolean",
    "string": "string",
}

def tools_to_json_schema(specs: list[dict]) -> list[dict]:
    """Convierte tools_spec.json al formato JSON Schema que pide el prompt."""
    schema_tools = []
    for tool in specs:
        properties = {
            p["name"]: {"type": TYPE_MAP.get(p["type"], "string")}
            for p in tool["parameters"]
        }
        required = [p["name"] for p in tool["parameters"] if p["required"]]

        schema_tools.append({
            "type": "function",
            "function": {
                "name":        tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required
                }
            }
        })
    return schema_tools


def build_lora_dataset(input_folder, output_file: str = "train_lora.jsonl"):
    tools_list    = tools_to_json_schema(TOOLS_SPEC)
    tool_names    = {t["name"] for t in TOOLS_SPEC}

    system_prompt = (
        f"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags."
        f"You have to call one function to assist with the user query. Don't make assumptions about what values to plug into functions"
        f"Here are the available tools:<tools> {json.dumps(tools_list)} </tools>"
        f"For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
        f"<tool_call>\n{{tool_call}}\n</tool_call>"
    )

    final_data = []

    for tool in TOOLS_SPEC:
        file_path = os.path.join(input_folder, f"{tool['name']}.json")
        if not os.path.exists(file_path):
            print(f"⚠️  No encontrado: {file_path}, saltando...")
            continue

        with open(file_path, "r") as f:
            data_dict = json.load(f)

        if isinstance(data_dict, dict) and "prompts" in data_dict:
            samples = data_dict["prompts"]
        elif isinstance(data_dict, list):
            samples = data_dict
        else:
            print(f"⚠️  Formato inesperado en {tool['name']}, saltando...")
            continue

        for s in samples:
            query            = s.get("user_query", "")
            tool_call_payload = {
                "name":      s.get("tool_call", tool["name"]),
                "arguments": s.get("arguments", {})
            }

            human_text = f"<bos><start_of_turn>human\n{system_prompt}\n\n{query}<end_of_turn><eos>"
            model_text = f"\n<start_of_turn>model\n<tool_call>\n{json.dumps(tool_call_payload)}\n</tool_call><end_of_turn><eos>"

            final_data.append({"text": human_text + model_text})

    random.shuffle(final_data)

    with open(output_file, "w") as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")

    print(f" Transformación completada: {len(final_data)} ejemplos listos en {output_file}")


if __name__ == "__main__":
    build_lora_dataset(input_folder="dataset_train_all_tools", output_file="train_lora_all_tools.jsonl")