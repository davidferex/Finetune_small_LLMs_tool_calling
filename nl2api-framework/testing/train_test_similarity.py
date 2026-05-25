import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
TRAIN_FILE = "./datasets/dataset_train/full_training_dataset.json"
TEST_FILE = "./datasets/dataset_test/full_testing_dataset.json"

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
# TOOL LIST
# ----------------------------------------------------------
tools = sorted(set(train_df[TOOL_COL]) | set(test_df[TOOL_COL]))

results = []

# ----------------------------------------------------------
# TOOL-BY-TOOL ANALYSIS
# ----------------------------------------------------------
print("\nComputing calibrated similarities...\n")

for tool in tools:

    train_mask = train_df[TOOL_COL] == tool
    test_mask  = test_df[TOOL_COL] == tool

    train_idx = train_df[train_mask].index.values
    test_idx  = test_df[test_mask].index.values

    if len(train_idx) == 0 or len(test_idx) == 0:
        continue

    train_vectors = train_emb[train_idx]
    test_vectors  = test_emb[test_idx]

    # ======================================================
    # TRAIN vs TEST
    # ======================================================
    train_test_matrix = np.dot(test_vectors, train_vectors.T)

    train_test_mean = train_test_matrix.mean()

    # ======================================================
    # TRAIN vs TRAIN
    # ======================================================
    train_train_matrix = np.dot(train_vectors, train_vectors.T)

    # Remove diagonal (self-similarity = 1)
    train_train_matrix = train_train_matrix[
        ~np.eye(train_train_matrix.shape[0], dtype=bool)
    ]

    train_train_mean = train_train_matrix.mean()

    # ======================================================
    # TEST vs TEST
    # ======================================================
    test_test_matrix = np.dot(test_vectors, test_vectors.T)

    # Remove diagonal
    test_test_matrix = test_test_matrix[
        ~np.eye(test_test_matrix.shape[0], dtype=bool)
    ]

    test_test_mean = test_test_matrix.mean()

    # ======================================================
    # GAINS
    # ======================================================
    gain_train = train_test_mean / train_train_mean
    gain_test  = train_test_mean / test_test_mean

    # ======================================================
    # SAVE
    # ======================================================
    results.append({
        "tool_call": tool,

        "train_test_similarity": round(float(train_test_mean), 4),
        "train_train_similarity": round(float(train_train_mean), 4),
        "test_test_similarity": round(float(test_test_mean), 4),

        "gain_vs_train": round(float(gain_train), 4),
        "gain_vs_test": round(float(gain_test), 4),

        "n_train": len(train_idx),
        "n_test": len(test_idx)
    })

# ----------------------------------------------------------
# RESULTS
# ----------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n================ CALIBRATED OVERLAP =================\n")
print(results_df.to_string(index=False))

# ----------------------------------------------------------
# SAVE CSV
# ----------------------------------------------------------
results_df.to_csv("calibrated_similarity_analysis.csv", index=False)

print("\nSaved: calibrated_similarity_analysis.csv")