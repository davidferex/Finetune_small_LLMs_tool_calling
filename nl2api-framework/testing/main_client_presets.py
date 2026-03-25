import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm_adapter import LLMAdapter

############################################
# CONFIG
############################################

INPUT_DATASET = "test.json"      # tu dataset original
OUTPUT_EVAL_FILE = "results_LLAMA.json"      # el que usa evaluate()

SYSTEM_PROMPT = """
You are a specialized Data Simulation Agent. Your primary responsibility is to act as an orchestrator between a user's natural language requirements and a set of technical data generation presets.

Based on the user's request, you must decide which specific preset tool is the most appropriate. You must also intelligently adjust the parameters to best match the user's intent.

Answer ONLY with the tool call and parameters using your tool_calls section.

If information is missing for the parameters use the tool 'MoreInformationNeeded'.

Use only the provided tools.
"""

############################################
# MAIN PIPELINE
############################################

async def run_evaluation():

    # Load dataset
    with open(INPUT_DATASET, "r") as f:
        dataset = json.load(f)

    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "main_server_presets.py"
    )

    params = StdioServerParameters(
        command=sys.executable,
        args=[server_script]
    )

    results = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            adapter = LLMAdapter()
            mcp_tools = await session.list_tools()
            adapter.load_tools(mcp_tools.tools)

            print(f"🧠 Using model: {adapter.model_name}")
            print(f"📊 Evaluating {len(dataset)} samples...\n")

            for i, sample in enumerate(dataset):

                user_query = sample["user_query"]
                true_tool = sample["tool_call"]
                true_args = sample["arguments"]

                messages = [
                    {"role": "user", "content": user_query}
                ]

                try:
                    prediction = await adapter.chat(
                        messages=messages,
                        system_prompt=SYSTEM_PROMPT
                    )

                    pred_tool = prediction.get("tool", None)
                    pred_args = prediction.get("params", {})

                except Exception as e:
                    print(f"❌ Error at sample {i}: {e}")
                    pred_tool = None
                    pred_args = {}

                results.append({
                    "tool": true_tool,
                    "arguments": true_args,
                    "prediction": {
                        "tool": pred_tool,
                        "arguments": pred_args
                    }
                })

                print(f"[{i+1}/{len(dataset)}] Done")

    # Save evaluation file
    with open(OUTPUT_EVAL_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation file generated:", OUTPUT_EVAL_FILE)


############################################
# ENTRY POINT
############################################

if __name__ == "__main__":

    if not os.path.exists("mcp_config.env"):
        print("❌ Error: No se encuentra mcp_config.env")
        sys.exit(1)

    load_dotenv("mcp_config.env")

    asyncio.run(run_evaluation())