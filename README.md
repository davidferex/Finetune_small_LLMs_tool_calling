# Fine-Tuning Small LLMs for Tool Calling in Bioinformatics APIs

This repository contains the work developed for my Master's final project, focused on fine-tuning small Large Language Models (LLMs) for tool calling and Natural Language to API (NL2API) tasks within bioinformatics environments.

The main objective of this work is to study whether relatively small open-source models, specifically Gemma 2, can learn to interact with a specialized API through supervised fine-tuning and synthetic dataset generation.

The repository is divided into two main directories:

- `nl2api-framework/` → Main contribution of the work. Contains the complete framework for dataset generation, fine-tuning, evaluation, and experimentation with tool-calling models.
- `WGCNA/` → Preliminary exploratory work used as an initial proof of concept for orchestrating bioinformatics workflows with LLMs and MCP configurations.

---

# Repository Structure

## `nl2api-framework/`

This directory contains the core contribution of the work.

It includes:

- Synthetic dataset generation pipelines for NL2API tasks
- API and tool schema definitions
- Fine-tuning pipelines for Gemma 2 models
- Tool-calling evaluation and benchmarking scripts
- Incomplete and ambiguous query generation
- Experiments
- Utilities for inference and structured output generation

The framework is specifically designed to evaluate how small LLMs can learn to generate valid API calls from natural language instructions in scientific domains.

The API used throughout the experiments focuses on bioinformatics workflows and computational biology tools.

---

## `WGCNA/`

This directory contains an earlier exploratory prototype developed before the main framework.

Its purpose was to experiment with:

- LLM-based orchestration of scientific workflows with MCP
- Tool calling in bioinformatics environments
- Automation of WGCNA (Weighted Gene Co-expression Network Analysis) pipelines
- Integration between language models and computational biology tools

Although functional, this directory should be considered a preliminary proof of concept and first contact with the problem domain rather than the main contribution of the work.

---

# Technologies Used

Some of the technologies and tools used throughout the project include:

- Python
- Hugging Face Transformers
- Gemma 2
- Ollama
- FastMCP
- JSON Schema
- PEFT / LoRA fine-tuning
- Bioinformatics tooling for computational workflows

---

# Project Context

This repository was developed as part of a Master's final project in Artificial Intelligence and Computer Engineering.

The work explores how small open-source language models can be adapted for structured tool calling tasks in specialized scientific APIs, with a particular focus on bioinformatics applications.

Special attention is given to synthetic dataset generation, robustness against incomplete queries, and efficient fine-tuning strategies for resource-constrained models.

---

# Rights and Usage

This repository is shared for academic, educational, and research purposes.

Unless explicitly stated otherwise:

- The code remains the intellectual property of the author
- Redistribution or commercial use may require prior permission
- External libraries, models, and datasets used in this repository are subject to their own respective licenses
- Generated datasets and experimental artifacts may depend on third-party tools and models

Please cite the repository and/or the associated final document if you use this work in academic research.
