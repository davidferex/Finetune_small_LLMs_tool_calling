import asyncio
import os
import sys
import json
import ast
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llm_adapter import LLMAdapter
# Asumiendo que las clases anteriores están en prompts.py
from prompts import FilterPrompt, NormalizePrompt

# Mapeo de estados a configuración
STATE_CONFIG = {
    "FILTERING": {
        "class": FilterPrompt,
        "tools": ["FILTER"],
        "next": "NORMALIZING"
    },
    "NORMALIZING": {
        "class": NormalizePrompt,
        "tools": ["NORM"],
        "next": "DONE"
    }
}

async def run_pipeline():
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_server.py")
    params = StdioServerParameters(command=sys.executable, args=[server_script])

    research_question = "Quiero identificar los 100 genes más informativos para ver patrones claros."

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            adapter = LLMAdapter("AIzaSyBYchdpqDFhmZ1L3-fDiEWq6uV-aAmw55E") 
            mcp_tools = await session.list_tools()
            adapter.load_tools(mcp_tools.tools)

            print("\n🧬 --- SISTEMA DE PROCESAMIENTO AUTÓNOMO (Graph-CoT) ---")
            
            current_state = "FILTERING"
            # Inicializamos con la pregunta de investigación
            messages = [{"role": "user", "content": f"Peticion: {research_question}. Por favor, aplica el Data Wrangling necesario."}]
            pipeline_active = True
            pasos = 0
            
            while pipeline_active and pasos < 6:
                pasos += 1
                
                stats_output = await session.call_tool("Get_data_statistics", {})
                # 1. OBTENCIÓN AUTOMÁTICA DE ESTADÍSTICAS (El "Sensor")
                # Llamamos a la herramienta de stats sin que el LLM lo pida
                stats_dict = stats_output.content[0].text if hasattr(stats_output.content[0], 'text') else stats_output
                # Intentamos convertir la respuesta (que viene como string de dict) a dict real
                try:
                    actual_stats = ast.literal_eval(stats_dict) if isinstance(stats_dict, str) else stats_dict
                except:
                    actual_stats = {}

                # 2. SELECCIÓN DE PROMPT Y HERRAMIENTAS SEGÚN ESTADO
                config = STATE_CONFIG[current_state]
                prompt_obj = config["class"](stats=actual_stats)
                system_prompt = prompt_obj.get_content(research_question)
                
                # Restringimos herramientas visibles para el LLM en este estado
                adapter.set_active_tools(config["tools"])

                print(f"\n🧠 [Paso {pasos} - Estado: {current_state}] Pensando...")

                # 3. LLAMADA AL LLM
                resultado = await adapter.chat(
                    messages=messages,
                    system_prompt=system_prompt
                )

                if resultado["tool"]:
                    print(f"🛠️  Acción elegida: {resultado['tool']}")
                    
                    # Ejecución en el servidor
                    tool_output = await session.call_tool(resultado["tool"], resultado["params"])
                    observacion = tool_output.content[0].text
                    print(f"✅ Resultado: {observacion}")

                    # Guardar en memoria
                    messages.append({"role": "assistant", "content": f"Ejecuté {resultado['tool']} con parámetros {resultado['params']}"})
                    messages.append({"role": "user", "content": f"Resultado: {observacion}. ¿Cómo procedemos?"})
                    
                    # 4. TRANSICIÓN DE ESTADO
                    current_state = config["next"]
                    if current_state == "DONE":
                        pipeline_active = False
                    
                    await asyncio.sleep(1)
                else:
                    print(f"🏁 El modelo ha finalizado el proceso.")
                    print(f"Respuesta final: {resultado['content']}")
                    pipeline_active = False

            print("\n✨ --- PIPELINE COMPLETADO SEGÚN ESTADOS ---")

if __name__ == "__main__":
    asyncio.run(run_pipeline())