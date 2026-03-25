import json
import requests
import os
import random
import re
from dotenv import load_dotenv
from ollama import Client

load_dotenv("mcp_config.env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

TOOLS_SPEC = [
    {"name": "FastPrototypePreset", "description": "Preset optimized for rapid data generation. Ideal for integration testing, CI/CD pipelines, or quick prototyping.", "parameters": ["n_samples (int)"]},
    {"name": "HighFidelityPreset", "description": "Preset optimized for maximum data quality and fidelity. Uses CTGAN with a high number of epochs and optimized batch size, prioritizing quality over speed. Forces adversarial validation.", "parameters": ["n_samples (int)", "auto_report (bool) (default)"]},
    {"name": "ImbalancedGeneratorPreset", "description": "Preset designed to generate synthetic data with a specific imbalanced distribution. Useful for creating test datasets for drift detection or bias analysis. Uses 'resample' or generative methods with forced custom distributions on the target.", "parameters": ["n_samples (int)", "target_col (string)", "imbalance_ratio (float) (default)"]},
    {"name": "TimeSeriesPreset", "description": "Preset optimized for generating sequential or time-series data. Uses TimeGAN, TimeVAE, or FourierFlows (fflows) to capture temporal dynamics. TimeGAN: best for complex temporal patterns. TimeVAE: faster, good for regular series. fflows: most stable, best for periodic/seasonal series", "parameters": ["n_samples (int)", "sequence_key (string) (default)", "time_col (string)", "method (string) (default)"]},
    {"name": "BalancedDataGeneratorPreset", "description": "Preset designed to balance an originally imbalanced dataset. Uses SMOTE (or ADASYN) to oversample minority classes to achieve a balanced distribution.", "parameters": ["n_samples (int)", "target_col (string)"]}
]

def prepare_params(params, drop_prob=0.3):
    clean_params = []
    default_params = []

    for p in params:
        is_default = "(default)" in p
        
        # quitar (default)
        p_clean = p.replace("(default)", "").strip()

        if is_default:
            default_params.append(p_clean)
        else:
            clean_params.append(p_clean)

    # decidir si eliminamos defaults
    final_defaults = []
    for p in default_params:
        if random.random() > drop_prob:
            final_defaults.append(p)

    return clean_params + final_defaults


def remove_params(params, min_remove=1, max_remove=None):
    default_params = [p for p in params if "(default)" in p]
    non_default_params = [p for p in params if "(default)" not in p]

    def random_remove(param_list, min_r=0):
        if not param_list:
            return param_list
        max_r = min(max_remove or len(param_list), len(param_list))
        n_remove = random.randint(min_r, max_r)
        to_remove = set(random.sample(param_list, n_remove))
        return to_remove

    kept_non_default = random_remove(non_default_params, min_r=min_remove) if len(non_default_params) > 1 else non_default_params
    kept_default = random_remove(default_params, min_r=0)

    return kept_non_default + kept_default
        


def call_ollama(prompt):
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.9}
    }
    try:
        response = requests.post(url, json=payload, timeout=20000)
        response.raise_for_status()
        return json.loads(response.json().get('response', '{}'))
    except Exception as e:
        print(f"Error procesando JSON de Ollama: {e}")
        return None


def run_dataset_generation(n_samples_completas=50, split_incompleta=0.2):
    if not os.path.exists("dataset_raw2"):
        os.makedirs("dataset_raw2")

    final_dataset = []
    final_dataset_mlp = []

    for tool in TOOLS_SPEC:
        print(f"\n--- Pipeline: {tool['name']} ---")
        tool_results = []
        tool_results_mlp = []
        samples_count = 0
        last_query = "None yet"
        
        while samples_count < n_samples_completas:
            # PASO 1: Generar parámetros técnicos
            clean_param_list = prepare_params(tool['parameters'])
            prompt_params = f"Generate a JSON list of 5 sets of parameters for '{tool['name']}'. Params: {clean_param_list}. Output example: [{{'n_samples': 100}}]"
            res = call_ollama(prompt_params)
            
            # Lógica para extraer la lista sin importar si viene envuelta en una clave
            params_list = []
            if isinstance(res, list):
                params_list = res
            elif isinstance(res, dict):
                # Buscamos la primera lista que haya dentro del diccionario
                for val in res.values():
                    if isinstance(val, list):
                        params_list = val
                        break

            if not params_list:
                print("No se encontró lista de parámetros, reintentando...")
                continue

            for params in params_list:
                if samples_count >= n_samples_completas: break
                if not isinstance(params, dict): continue

                # PASO 2: Generar Query Completa
                prompt_query = f"Write a natural user request for: {params}. Context: {tool['description']}."
                if last_query != "None yet":
                    prompt_query += f" We need diversity. Change the query, the structure and tone from the last one: {last_query}."
                prompt_query += f" JSON: {{\"user_query\": \"text\"}}"
                query_obj = call_ollama(prompt_query)
                if not query_obj or "user_query" not in query_obj: continue
                user_query = query_obj["user_query"]
                last_query = user_query

                tool_results.append({
                    "user_query": user_query,
                    "tool_call": tool["name"],
                    "arguments": params
                })

                tool_results_mlp.append({
                    "user_query": user_query,
                    "complete": 1
                })

                samples_count += 1
                
                print(f"[{tool['name']}] {samples_count}/{n_samples_completas}")

                params_to_remove = remove_params(tool['parameters'])
                print(params_to_remove)
                # Generar Query Incompleta
                prompt_query = f"Make the following query incomplete by removing the mention or the value of {params_to_remove}. You have this query: {user_query}."
                prompt_query += f" JSON: {{\"user_query\": \"text\"}}"
                query_obj = call_ollama(prompt_query)
                if not query_obj or "user_query" not in query_obj: continue
                user_query = query_obj["user_query"]

                tool_results_mlp.append({
                    "user_query": user_query,
                    "complete": 0
                })
                                
                print(f"[{tool['name']}] {samples_count}/{n_samples_completas}")
        
        
        with open(f"dataset_raw2/{tool['name']}.json", "w") as f:
            json.dump(tool_results, f, indent=4)
        final_dataset.extend(tool_results)

        with open(f"dataset_raw2/{tool['name']}_mlp.json", "w") as f:
            json.dump(tool_results_mlp, f, indent=4)
        final_dataset_mlp.extend(tool_results_mlp)

    with open("dataset_raw2/full_training_dataset.json", "w") as f:
        json.dump(final_dataset, f, indent=4)

    with open("dataset_raw2/full_training_dataset_mlp.json", "w") as f:
        json.dump(final_dataset_mlp, f, indent=4)

if __name__ == "__main__":
    run_dataset_generation(50, 0.2)
