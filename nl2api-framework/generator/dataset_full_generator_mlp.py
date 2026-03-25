import json
import os
import random

# Tu fuente de verdad (reutilizada)
TOOLS_SPEC = [
    {"name": "FastPrototypePreset", "description": "Preset optimized for rapid data generation. Ideal for integration testing, CI/CD pipelines, or quick prototyping.", "parameters": ["n_samples (int)"]},
    {"name": "HighFidelityPreset", "description": "Preset optimized for maximum data quality and fidelity. Uses CTGAN with a high number of epochs and optimized batch size, prioritizing quality over speed. Forces adversarial validation.", "parameters": ["n_samples (int)", "auto_report (bool)"]},
    {"name": "ImbalancedGeneratorPreset", "description": "Preset designed to generate synthetic data with a specific imbalanced distribution. Useful for creating test datasets for drift detection or bias analysis. Uses 'resample' or generative methods with forced custom distributions on the target.", "parameters": ["n_samples (int)", "target_col (string)", "imbalance_ratio (float)"]},
    {"name": "TimeSeriesPreset", "description": "Preset optimized for generating sequential or time-series data. Uses TimeGAN, TimeVAE, or FourierFlows (fflows) to capture temporal dynamics. TimeGAN: best for complex temporal patterns. TimeVAE: faster, good for regular series. fflows: most stable, best for periodic/seasonal series", "parameters": ["n_samples (int)", "sequence_key (string)", "time_col (string)", "method (string)"]},
    {"name": "BalancedDataGeneratorPreset", "description": "Preset designed to balance an originally imbalanced dataset. Uses SMOTE (or ADASYN) to oversample minority classes to achieve a balanced distribution.", "parameters": ["n_samples (int)", "target_col (string)"]}
]

def tools_to_json_schema(specs):
    """Convierte tu TOOLS_SPEC al formato JSON Schema que pide el prompt"""
    schema_tools = []
    for tool in specs:
        properties = {}
        for param in tool["parameters"]:
            # Mapeo básico de tipos para el esquema
            p_type = "integer" if "n_samples" in param else "string"
            if "ratio" in param: p_type = "float"
            if "auto_report" in param: p_type = "boolean"
            
            properties[param] = {"type": p_type}
            
        schema_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": tool["parameters"]
                }
            }
        })
    return schema_tools

def build_lora_dataset(input_folder="dataset_raw2", output_file="train_mlp.jsonl"):
    tools_list = tools_to_json_schema(TOOLS_SPEC)
    
    # Template exacto que pediste
    system_prompt = (
        f"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags."
        f"You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions."
        f"Here are the available tools:<tools> {json.dumps(tools_list)} </tools>"
        f"Use the following pydantic model json schema for each tool call you will make: "
        f"{{'title': 'FunctionCall', 'type': 'object', 'properties': {{'arguments': {{'title': 'Arguments', 'type': 'object'}}, 'name': {{'title': 'Name', 'type': 'string'}}}}, 'required': ['arguments', 'name']}} "
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
    build_lora_dataset()