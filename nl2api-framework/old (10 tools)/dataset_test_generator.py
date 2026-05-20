import json
import requests
import os
import random
import re
import copy
from dotenv import load_dotenv

load_dotenv("mcp_config.env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Cargar spec desde fichero
with open("tools_spec.json", "r") as f:
    TOOLS_SPEC = json.load(f)


# Helpers de parámetros

def prepare_params(parameters: list[dict], drop_prob: float = 0.4) -> tuple[list[dict], list[dict]]:
    result = []
    dropped = []
    for p in parameters:
        include = p["required"] or (random.random() > drop_prob)
        if include:
            p_copy = copy.deepcopy(p)
            p_copy.pop("required", None)
            result.append(p_copy)
        else:
            p_copy = copy.deepcopy(p)
            p_copy.pop("required", None)
            p_copy.pop("default", None)
            dropped.append(p_copy)
    return result, dropped


def get_incomplete_strategy(tool: dict) -> tuple[list[str], list[str]]:
    required = [p["name"] for p in tool["parameters"] if p["required"]]
    default  = [p["name"] for p in tool["parameters"] if not p["required"]]

    def random_remove(param_list, min_r=0):
        if len(param_list) < 1:
            return [], []
        n_remove = random.randint(min_r, len(param_list))
        to_remove = set(random.sample(param_list, n_remove))
        return [p for p in param_list if p not in to_remove], [p for p in param_list if p in to_remove]

    kept_non_default, omit_non_default = random_remove(required, min_r=1)
    kept_default,     omit_default     = random_remove(default,   min_r=0)
    return kept_default + kept_non_default, omit_default + omit_non_default


def build_incomplete_prompt(user_query: str, keep: list[str], omit: list[str], tool: dict) -> str:
    omit_descriptions = {
        p["name"]: p["description"]
        for p in tool["parameters"]
        if p["name"] in omit
    }
    return (
        f"Rewrite the following query making it incomplete.\n\n"
        f"ORIGINAL QUERY: \"{user_query}\"\n"
        f"PARAMETERS TO KEEP (must remain clear): {keep}\n"
        f"PARAMETERS TO OMIT (must NOT appear or be inferable): {omit_descriptions}\n\n"
        f"Rules:\n"
        f"- Keep the same general intent and tone\n"
        f"- Do not add new information\n"
        f"- The result must sound like a real user who forgot to specify something\n"
        f"- If omitting a numeric param: use vague quantities like 'some', 'a few', 'several'\n"
        f"- If omitting a string param: do not name the column, key or method\n"
        f"- If omitting a float/ratio: do not give any numeric hint\n\n"
        f"Respond ONLY with JSON: {{\"user_query\": \"...\"}}"
    )


# Ollama

def call_ollama(prompt: str) -> dict | None:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 1.0}
    }
    try:
        response = requests.post(url, json=payload, timeout=20000)
        response.raise_for_status()
        raw = response.json().get("response", "{}")

        if "deepseek" in OLLAMA_MODEL.lower():
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        return json.loads(raw)
    except Exception as e:
        print(f"Error procesando JSON de Ollama: {e}")
        return None


STYLES = [
    "formal",
    "informal",
    "short",
    "verbose",
    "slightly ambiguous",
    "with typos",
    "spoken language",
    "non-native speaker"
]

# Pipeline principal

def run_dataset_generation(n_samples_completas: int = 50):
    os.makedirs("dataset_test", exist_ok=True)

    final_dataset          = []  # tool calling
    final_dataset_mlp      = []  # MLP (binario)

    use_default_string_prob = 0.3

    for tool in TOOLS_SPEC:
        print(f"\n--- Pipeline: {tool['name']} ---")
        tool_results          = []
        tool_results_mlp      = []
        samples_count = 0
        last_query = "None yet"

        while samples_count < n_samples_completas:

            # PASO 1: Generar parámetros técnicos
            param_names, dropped = prepare_params(tool["parameters"])
            prompt_params = (
                f"Generate a JSON list of 5 sets of parameters for '{tool['name']}'. "
                f"Params: {param_names}. "
                f"Output example: [{{'n_samples': 100}}]"
            )
            print(prompt_params)
            res = call_ollama(prompt_params)

            params_list = []
            if isinstance(res, list):
                params_list = res
            elif isinstance(res, dict):
                for val in res.values():
                    if isinstance(val, list):
                        params_list = val
                        break

            if not params_list:
                print("No se encontró lista de parámetros, reintentando...")
                continue

            dropped_defaults = {
                p["name"]: p["default"]
                for p in tool["parameters"]
                if not p["required"]
                and p["name"] in {d["name"] for d in dropped}
                and "default" in p
            }
            print(params_list)
            print(dropped_defaults)
 

            for params in params_list:
                if samples_count >= n_samples_completas:
                    break
                if not isinstance(params, dict):
                    continue
                
                params_with_defaults = copy.deepcopy(params)
                for param_name, default_value in dropped_defaults.items():
                    if param_name not in params_with_defaults:
                        params_with_defaults[param_name] = default_value

                style = random.choice(STYLES)
                # PASO 2: Generar Query Completa
                prompt_query = f"Write a natural user request for: {params}. "
                prompt_query += f" Write it in a {style} style."
                if len(dropped) > 0 and random.random() <= use_default_string_prob:
                    prompt_query += f"For the following parameters, mention to use the default value: {dropped}"
                prompt_query += f"Context: {tool['description']}."
                if last_query != "None yet":
                    prompt_query += f" We need diversity. Change the query, the structure and tone from the last one: {last_query}."
                prompt_query += f" JSON: {{\"user_query\": \"text\"}}"

                query_obj = call_ollama(prompt_query)
                if not query_obj or "user_query" not in query_obj:
                    continue
                user_query = query_obj["user_query"]
                last_query = user_query

                # Dataset tool calling
                tool_results.append({
                    "user_query": user_query,
                    "tool_call":  tool["name"],
                    "arguments":  params_with_defaults
                })

                # Dataset MLP — completo
                tool_results_mlp.append({
                    "user_query": user_query,
                    "complete":   1
                })

                samples_count += 1
                print(f"[{tool['name']}] {samples_count}/{n_samples_completas}")

                # PASO 3: Generar Query Incompleta
                keep, omit = get_incomplete_strategy(tool)
                print(f"  keep={keep} | omit={omit}")

                prompt_incomplete = build_incomplete_prompt(user_query, keep, omit, tool)
                incomplete_obj = call_ollama(prompt_incomplete)
                if not incomplete_obj or "user_query" not in incomplete_obj:
                    continue
                incomplete_query = incomplete_obj["user_query"]

                # Dataset MLP — incompleto
                tool_results_mlp.append({
                    "user_query": incomplete_query,
                    "complete":   0,
                    "_omitted":   omit
                })

                print(f"[{tool['name']}] incompleto generado (omitido: {omit})")

        # Guardar por tool
        with open(f"dataset_test/{tool['name']}.json", "w", encoding="utf-8") as f:
            json.dump(tool_results, f, indent=4, ensure_ascii=False)

        with open(f"dataset_test/{tool['name']}_mlp.json", "w", encoding="utf-8") as f:
            json.dump(tool_results_mlp, f, indent=4, ensure_ascii=False)

        final_dataset.extend(tool_results)
        final_dataset_mlp.extend(tool_results_mlp)

    # Guardar datasets globales
    with open("dataset_test/full_testing_dataset.json", "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)

    with open("dataset_test/full_testing_dataset_mlp.json", "w", encoding="utf-8") as f:
        json.dump(final_dataset_mlp, f, indent=4, ensure_ascii=False)

    n_mlp_completos   = sum(1 for e in final_dataset_mlp      if e["complete"] == 1)
    n_mlp_incompletos = sum(1 for e in final_dataset_mlp      if e["complete"] == 0)

    print(f"\n Dataset tool calling: {len(final_dataset)} ejemplos")
    print(f" Dataset MLP:          {len(final_dataset_mlp)} ejemplos ({n_mlp_completos} completos, {n_mlp_incompletos} incompletos)")

if __name__ == "__main__":
    run_dataset_generation(40)