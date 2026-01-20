import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llm_adapter import LLMAdapter
from orchestrator import BioOrchestrator
from pipeline_loader import PipelineLoader

async def main():
    # Configuración del servidor
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_server.py")
    params = StdioServerParameters(command=sys.executable, args=[server_script])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Inicializar IA
            adapter = LLMAdapter("AIzaSyBYchdpqDFhmZ1L3-fDiEWq6uV-aAmw55E") 
            mcp_tools = await session.list_tools()
            adapter.load_tools(mcp_tools.tools)

            # Creamos el orquestador inyectándole la configuración
            config = PipelineLoader.load("pipeline.json")
            orchestrator = BioOrchestrator(session, adapter, config)
            
            question = "Quiero identificar los 10 genes más informativos."
            await orchestrator.run(question)

if __name__ == "__main__":
    asyncio.run(main())