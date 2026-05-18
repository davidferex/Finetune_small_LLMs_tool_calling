# NL2API Framework 🚀

A framework for generating natural language interfaces for any API through fine-tuning efficient local models (Gemma 2).

## 💡 Concept

This framework enables any developer to move from a static JSON API definition to a conversational interface capable of:

1. **Intent-to-code mapping:** Translating natural language into JSON API calls.
2. **Ambiguity handling:** Automatically detecting missing parameters.
3. **Local execution:** Optimized to run on Gemma 2 2B.

---

## 🛠️ How it works

1. **Define your API:** Create a `tools_spec.json` file with your functions and parameters.
2. **Generate the Dataset:** The dataset generation pipeline uses a teacher model (LLaMA 3.3) to create thousands of diverse interactions.
3. **Fine-tune:** Train a smaller model so it learns *your* specific API, Gemma 2 2B in this case.
4. **Ready to use:** The trained model should perform strongly at tool calling for your specific API.

---

## 📁 Project Structure

The repository is organized into several key directories, each corresponding to a stage of the pipeline:

### `datasets/`

Contains the generated datasets used for training and evaluation.

* Includes both intermediate and final datasets (e.g., raw queries, `train_lora.json`, `train_mlp.json`, etc.).
* These datasets are produced by the generation pipeline and later consumed during fine-tuning and testing.

---

### `generator/`

Implements the full data generation pipeline.

This stage bridges the gap between a tool specification and a usable training dataset.

---

### `finetuning/`

Contains the training scripts for adapting models to the target API.

* LoRA fine-tuning scripts for Gemma 2 (tool calling task) and LLM verifier.
* MLP training scripts (classifier head for tool prediction)

These scripts take the datasets from `datasets/` and produce trained models ready for inference.

---

### `testing/`

Includes all evaluation and benchmarking scripts.

This directory allows direct comparison between fine-tuned small models and large baseline models.

---

### `results/`

Includes all the results files generated during experiments.


### IMPORTANT

Note that HuggingFace and Weights&Biases were used during the experiments, so these scripts use their libraries. For the correct execution of everything, an environment file with you HF and W&B tokens is needed.

---

## 🧪 Use Case: Synthetic Data Generator

We validated the framework by building an interface for a data generation API: **[CalmDataGenerator](https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator)**

* **Input:** "I need 100 rows of a time series quickly."
* **Output:** `TimeSeriesPreset(n_samples=100, ...)`

Results show that the fine-tuned model can outperform larger non-fine-tuned models on the specific task of tool calling for the target API. In this case, it was compared against LLaMA 3.3 70B.

---

## 🧾 Summary

The full pipeline looks like this:

1. **Define tools** → `tools_spec.json`
2. **Generate data** → `generator/`
3. **Store datasets** → `datasets/`
4. **Train models** → `finetuning/`
5. **Evaluate & compare** → `testing/`

This modular structure allows you to easily adapt the framework to new APIs, datasets, and evaluation setups while keeping each stage cleanly separated.


