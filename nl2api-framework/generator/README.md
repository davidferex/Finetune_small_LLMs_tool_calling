# Data Generation Pipeline

This folder contains the full data generation pipeline used in the project for training Gemma 2 in tool calling tasks. The process is divided into two main stages: generating raw queries and transforming them into the final training format.

---

## Overview

The dataset generation follows a two-step approach:

1. **Query Generation**
   Synthetic queries are created using our pipeline featuring LLaMA 3.3 70B.

2. **Dataset Formatting**
   These queries are transformed into the final JSON format required for training (LoRA or MLP variants).

---

## Step 1: Query Generation

**File:** `dataset_preset_generator.py`

This script is responsible for generating the base dataset of queries. It takes de info from the API given through the spec JSON file and generates queries for each tool.

### What it does:

* Generates natural language queries
* Associates them with tool calls and parameters
* Produces an intermediate dataset (raw format)

### Output:

A JSON file containing:

* User queries
* Tool names
* Arguments

### Notes:

* You may need to adjust:

  * Output file name
  * Number of samples
  * Spec JSON file
* Ensure the generated queries align with the tools you plan to use later

---

## Step 2: Dataset Formatting

There are two main scripts depending on the training approach:

### 2.1 LoRA Format

**File:** `dataset_full_generator.py`

Transforms the raw dataset into a format suitable for LoRA fine-tuning.

### What it does:

* Converts queries into instruction-style prompts
* Formats tool calls into structured outputs

### Output:

`train_lora.json`

---

### 2.2 MLP Format

**File:** `dataset_full_generator_mlp.py`

Generates a dataset tailored for training a classifier (MLP head).

### Output:

`train_mlp.json`

---

## Important Considerations

### File Paths and Naming

Some scripts are not fully parameterized and may require manual adjustments:

* Input dataset path
* Output file names (`train_lora.json`, `train_mlp.json`, etc.)
* Spec JSON file

Make sure paths are consistent across scripts.

---

### Tool Specifications

The generated dataset depends heavily on the tool specification used.

* If you change the tools:

  * Update the tool schema/spec file
  * Ensure consistency with the generator logic
* Mismatches between queries and tool definitions will lead to incorrect training data

## Typical Workflow

1. Run:

   ```bash
   python dataset_preset_generator.py
   ```

2. Then choose one:

   * For LoRA:

     ```bash
     python dataset_full_generator.py
     ```
   * For MLP:

     ```bash
     python dataset_full_generator_mlp.py
     ```

---

## Summary

* `dataset_preset_generator.py` → Generates raw queries
* `dataset_full_generator.py` → Builds LoRA training dataset
* `dataset_full_generator_mlp.py` → Builds MLP training dataset

Each step is modular, but requires careful alignment in terms of file paths, tool specifications, and expected formats.

---

If something breaks, the first thing to check is:

* File paths
* Tool spec consistency
* Expected input/output formats between scripts

This pipeline is flexible, but not fully automated — some manual adjustments are expected depending on the experiment setup.

