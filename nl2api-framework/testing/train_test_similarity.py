# ==========================================================
# Train-Test Similarity per Tool (Global + Pairwise Mean)
# ==========================================================
# pip install pandas numpy sentence-transformers tqdm
# ==========================================================

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
TRAIN_FILE = "./dataset_train_all_tools/full_training_dataset.json"
TEST_FILE = "./dataset_test_all_tools/full_testing_dataset.json"

TEXT_COL = "user_query"
TOOL_COL = "tool_call"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df[TEXT_COL] = train_df[TEXT_COL].astype(str)
test_df[TEXT_COL] = test_df[TEXT_COL].astype(str)

train_df[TOOL_COL] = train_df[TOOL_COL].astype(str)
test_df[TOOL_COL] = test_df[TOOL_COL].astype(str)

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ----------------------------------------------------------
# EMBEDDINGS
# ----------------------------------------------------------
print("Encoding train...")
train_emb = model.encode(
    train_df[TEXT_COL].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Encoding test...")
test_emb = model.encode(
    test_df[TEXT_COL].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

train_emb = np.array(train_emb)
test_emb = np.array(test_emb)

# ----------------------------------------------------------
# INDEX BY TOOL
# ----------------------------------------------------------
train_df["idx"] = range(len(train_df))
test_df["idx"] = range(len(test_df))

tools = sorted(set(train_df[TOOL_COL]) | set(test_df[TOOL_COL]))

results = []

# ----------------------------------------------------------
# TOOL-BY-TOOL SIMILARITY
# ----------------------------------------------------------
print("\nComputing tool-level similarities...\n")

for tool in tools:
    train_mask = train_df[TOOL_COL] == tool
    test_mask  = test_df[TOOL_COL] == tool

    train_idx = train_df[train_mask].index.values
    test_idx  = test_df[test_mask].index.values

    if len(train_idx) == 0 or len(test_idx) == 0:
        continue

    train_vectors = train_emb[train_idx]
    test_vectors  = test_emb[test_idx]

    # cosine similarity matrix (test x train)
    sim_matrix = np.dot(test_vectors, train_vectors.T)

    mean_sim = sim_matrix.mean()
    max_sim  = sim_matrix.max()
    min_sim  = sim_matrix.min()

    results.append({
        "tool_call": tool,
        "mean_similarity": mean_sim,
        "max_similarity": max_sim,
        "min_similarity": min_sim,
        "n_train": len(train_idx),
        "n_test": len(test_idx)
    })

# ----------------------------------------------------------
# GLOBAL SIMILARITY
# ----------------------------------------------------------
print("\nComputing global similarity...")

global_sim_matrix = np.dot(test_emb, train_emb.T)

global_mean = global_sim_matrix.mean()
global_max = global_sim_matrix.max()
global_min = global_sim_matrix.min()

print("\n================ GLOBAL =================")
print(f"Mean similarity : {global_mean:.4f}")
print(f"Max similarity  : {global_max:.4f}")
print(f"Min similarity  : {global_min:.4f}")

# ----------------------------------------------------------
# RESULTS TABLE
# ----------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("mean_similarity", ascending=False)

print("\n================ BY TOOL =================\n")
print(results_df.to_string(index=False))

# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------
results_df.to_csv("tool_train_test_similarity.csv", index=False)

print("\nSaved: tool_train_test_similarity.csv")