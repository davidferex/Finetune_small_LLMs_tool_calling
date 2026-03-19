# Bio-WGCNA Orchestrator (MCP + Ollama)

Este repositorio contiene un framework de orquestación de agentes inteligentes diseñado para automatizar el análisis de co-expresión génica (**WGCNA**). Utiliza el **Model Context Protocol (MCP)** para desacoplar la lógica del razonamiento (LLM) de la ejecución técnica de herramientas bioinformáticas.

## 🧬 Arquitectura del Sistema

El sistema utiliza una arquitectura tripartita diseñada para entornos de alto rendimiento (HPC):

*   **Cerebro (LLM):** Modelos de gran escala (`Llama 3.3 70B` / `Qwen 2.5 72B`) ejecutándose en **NVIDIA A100** mediante **Ollama**.
*   **Protocolo (MCP):** Capa de abstracción que permite al modelo listar, entender y ejecutar funciones de análisis de datos sin conocer la implementación interna.
*   **Ejecutor (Server):** Entorno de ejecución en Python/R que procesa grandes matrices de expresión de RNA-seq.

## 🚀 Requisitos y Configuración

### Dependencias
```bash
pip install mcp ollama python-dotenv pandas numpy
