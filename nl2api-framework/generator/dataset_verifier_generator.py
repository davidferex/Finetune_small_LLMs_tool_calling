import json
import random
import copy

# ==========================================================
# CONFIG
# ==========================================================
# Las rutas de los archivos deben ser modificadas según sea test o train.
TOOLCALLS_JSON = "dataset_train/full_training_dataset.json"   # json con tool calls correctas
MLP_JSON       = "dataset_train/full_training_dataset_mlp.json"         # json queries completas/incompletas
OUTPUT_JSON    = "verifier_dataset_train.json"

SEED = 42
random.seed(SEED)


# ==========================================================
# HELPERS
# ==========================================================

def mutate_value(value):
    """
    Genera un valor incorrecto distinto del original.
    """
    if isinstance(value, bool):
        return not value

    if isinstance(value, int):
        candidates = [
            value + 1,
            value + 5,
            value + 10,
            max(1, value - 1),
            max(1, value * 2),
        ]
        candidates = [x for x in set(candidates) if x != value]
        return random.choice(candidates)

    if isinstance(value, float):
        return round(value + random.uniform(0.5, 2.0), 3)

    if isinstance(value, str):
        if value.isdigit():
            return str(int(value) + 1)
        return value + "_wrong"

    return value


def choose_num_errors(n_params):
    """
    Número de parámetros erróneos:
    favorece 1, pero también 2,3...
    """
    if n_params == 1:
        return 1

    probs = []

    # pesos decrecientes
    for k in range(1, n_params + 1):
        probs.append(1 / k)

    total = sum(probs)
    probs = [p / total for p in probs]

    return random.choices(range(1, n_params + 1), weights=probs, k=1)[0]


def mutate_arguments(arguments):
    """
    Cambia 1..N parámetros aleatorios.
    """
    new_args = copy.deepcopy(arguments)
    keys = list(new_args.keys())

    if not keys:
        return new_args, []

    n_wrong = choose_num_errors(len(keys))
    chosen = random.sample(keys, n_wrong)

    for k in chosen:
        new_args[k] = mutate_value(new_args[k])

    return new_args, chosen


def hallucinate_missing_arguments(correct_args, omitted):
    """
    Mete valores inventados en parámetros omitidos.
    Además opcionalmente altera otros parámetros.
    """
    new_args = copy.deepcopy(correct_args)

    wrong_params = []

    # Siempre los omitidos están mal si aparecen
    for k in omitted:
        if k in new_args:
            new_args[k] = mutate_value(new_args[k])
            wrong_params.append(k)

    # Opcional: añadir más errores además de omitidos
    remaining = [k for k in new_args.keys() if k not in omitted]

    if remaining:
        extra_max = min(len(remaining), 2)  # hasta 2 extras
        extra_n = random.randint(0, extra_max)

        if extra_n > 0:
            extra_keys = random.sample(remaining, extra_n)

            for k in extra_keys:
                new_args[k] = mutate_value(new_args[k])
                wrong_params.append(k)

    return new_args, wrong_params


# ==========================================================
# LOAD DATA
# ==========================================================

with open(TOOLCALLS_JSON, "r", encoding="utf-8") as f:
    toolcalls = json.load(f)

with open(MLP_JSON, "r", encoding="utf-8") as f:
    mlp = json.load(f)

results = []


# ==========================================================
# MAIN LOOP
# ==========================================================

for i, tc in enumerate(toolcalls):

    mlp_complete   = mlp[2 * i]
    mlp_incomplete = mlp[2 * i + 1]

    query_good = tc["user_query"]
    tool_name  = tc["tool_call"]
    args_good  = tc["arguments"]

    # ------------------------------------------------------
    # 1) CORRECTO
    # ------------------------------------------------------
    results.append({
        "user_query": query_good,
        "tool_call": tool_name,
        "arguments": args_good,
        "label": 1,
        "error_type": "correct",
        "wrong_parameters": []
    })

    # ------------------------------------------------------
    # 2) MALOS VALORES (1..N params)
    # ------------------------------------------------------
    bad_args, wrong_params = mutate_arguments(args_good)

    results.append({
        "user_query": query_good,
        "tool_call": tool_name,
        "arguments": bad_args,
        "label": 0,
        "error_type": "wrong_argument_value",
        "wrong_parameters": wrong_params
    })

    # ------------------------------------------------------
    # 3) QUERY INCOMPLETA + TOOL INVENTA VALORES
    # ------------------------------------------------------
    omitted = mlp_incomplete.get("_omitted", [])

    hallucinated_args, wrong_params = hallucinate_missing_arguments(
        args_good,
        omitted
    )

    results.append({
        "user_query": mlp_incomplete["user_query"],
        "tool_call": tool_name,
        "arguments": hallucinated_args,
        "label": 0,
        "error_type": "value_not_present_in_query",
        "wrong_parameters": wrong_params
    })


# ==========================================================
# SAVE
# ==========================================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Generados {len(results)} ejemplos en {OUTPUT_JSON}")