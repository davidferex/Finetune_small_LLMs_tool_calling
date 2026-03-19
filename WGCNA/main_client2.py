import asyncio
import os
import sys
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Importamos tu nuevo adaptador y el orquestador
from llm_adapter import LLMAdapter
from orchestrator import BioOrchestrator
from pipeline_loader import PipelineLoader

# Cargamos la configuración del archivo que creamos para evitar conflictos con la carpeta .env
load_dotenv(dotenv_path="mcp_config.env")

async def main():
    # 1. Configuración del servidor MCP (se mantiene igual)
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_server.py")
    params = StdioServerParameters(command=sys.executable, args=[server_script])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 2. Inicializar conexión MCP
            await session.initialize()
            
            # 3. Inicializar el Adaptador de Ollama (Ya no necesita API Key)
            # El modelo y la IP los tomará automáticamente del mcp_config.env
            adapter = LLMAdapter() 
            
            # 4. Cargar herramientas en el adaptador
            mcp_tools = await session.list_tools()
            adapter.load_tools(mcp_tools.tools)

            # 5. Configurar el orquestador
            # Asegúrate de que 'pipeline.json' y tus prompts existan en la carpeta
            config = PipelineLoader.load("pipeline.json", "prompts")
            orchestrator = BioOrchestrator(session, adapter, config)
            
            # 6. Definir el objetivo del análisis
            question = """
            ROLE: Senior Bioinformatics Scientist (WGCNA Specialist).

            GOAL: Identify robust co-expression modules and hub genes with high biological significance.

            OPERATING PRINCIPLES:
            1. Choose one of the tools provided each time and only answer with the tool call.
            2. Always follow the indications provided by the user.
            """            
            
            print(f"🚀 Iniciando Pipeline en el servidor Eowyn...")
            print(f"🧠 Usando modelo: {adapter.model_name}")
            
            # 7. Ejecutar
            await orchestrator.run(question)

if __name__ == "__main__":
    # Verificación de seguridad para la conexión SSH/Red
    if not os.path.exists("mcp_config.env"):
        print("❌ Error: No se encuentra el archivo mcp_config.env")
        sys.exit(1)
        
    asyncio.run(main())