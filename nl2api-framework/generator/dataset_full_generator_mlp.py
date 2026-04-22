import json
import os
import random

with open("tools_spec.json", "r") as f:
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

def build_lora_dataset(input_folder, output_file="train_mlp.jsonl"):
    tools_list = tools_to_json_schema(TOOLS_SPEC)
    
    # Template exacto que pediste
    system_prompt = (
            f"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags."
            f"You have to call one function to assist with the user query. Don't make assumptions about what values to plug into functions"
            f"Here are the available tools:<tools> {json.dumps(tools_list)} </tools>"
            f"For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
            f"<tool_call>\n{{tool_call}}\n</tool_call>"
        )

    final_data = []

    for tool in TOOLS_SPEC:
        file_path = os.path.join(input_folder, f"{tool['name']}_mlp.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
            
            # Si es lista, usamos directamente; si es dict con 'prompts', usamos esa lista
            if isinstance(data_dict, dict) and "prompts" in data_dict:
                samples = data_dict["prompts"]
            elif isinstance(data_dict, list):
                samples = data_dict
            else:
                print(f"⚠️ Formato inesperado en {tool['name']}. Saltando...")
                continue

            for s in samples:
                # Extraer Query
                query = s.get("user_query", "")
                
                # Lógica de etiquetado: 
                # Si el JSON viene de tu generación de "casos incompletos", usará su flag.
                # Si no existe, por defecto asumimos que es 1 (completa).
                label = s.get("complete", "")
                
                # IMPORTANTE: No pongas <eos> al final. 
                # El MLP suele leer el hidden state del último token generado por el humano.
                human_text = f"<bos><start_of_turn>human\n{system_prompt}\n\n{query}<end_of_turn>"
                
                final_data.append({
                    "text": human_text,
                    "label": label,
                    "tool_target": tool['name'] # Útil para analítica posterior
                })
                
    random.shuffle(final_data)

    with open(output_file, 'w') as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Transformación completada: {len(final_data)} ejemplos listos.")

if __name__ == "__main__":
    build_lora_dataset(input_folder="dataset_test2", output_file="test_mlp.jsonl")