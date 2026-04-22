# NL2API Framework 🚀

A framework for generating natural language interfaces for any API through fine-tuning efficient local models (Gemma 2).

## 💡 Concept

This framework enables any developer to move from a static JSON API definition to a conversational interface capable of:

1. **Intent-to-code mapping:** Translating natural language into JSON API calls.
2. **Ambiguity handling:** Automatically detecting missing parameters.
3. **Local execution:** Optimized to run on Gemma 2 2B.

## 🛠️ How it works

1. **Define your API:** Create a `tools_spec.json` file with your functions and parameters.
2. **Generate the Dataset:** The `DatasetGenerator` uses a teacher model (LLaMA 3.3) to create thousands of diverse interactions.
3. **Fine-tune:** Train a smaller model so it learns *your* specific API.
4. **Ready to use:** The trained model should perform strongly at tool calling for your specific API.

## 🧪 Use Case: Synthetic Data Generator

We validated the framework by building an interface for a data generation API: **[CalmDataGenerator](https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator)**

* **Input:** "I need 100 rows of a time series quickly."
* **Output:** `TimeSeriesPreset(n_samples=100, ...)`

Results show that the fine-tuned model can outperform larger non-fine-tuned models on the specific task of tool calling for the target API. In this case, it was compared against LLaMA 3.3 70B.

