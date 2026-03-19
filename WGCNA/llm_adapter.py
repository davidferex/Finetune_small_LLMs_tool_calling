import os
import logging
import asyncio
from ollama import AsyncClient
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

# Cargamos el archivo con el nuevo nombre para evitar conflictos con la carpeta .env
load_dotenv(dotenv_path="mcp_config.env")

logger = logging.getLogger(__name__)

class LLMAdapter:
    def __init__(
        self,
        model: Optional[str] = None
    ):
        # Cargamos configuración desde mcp_config.env
        self.host = os.getenv("OLLAMA_HOST", "http://155.54.95.92:11434")
        self.model_name = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
        
        # Inicializamos el cliente asíncrono de Ollama
        self.client = AsyncClient(host=self.host)
        
        self.tools = []
        self._tools_by_tag = {}
        self.active_tools = []

    def load_tools(self, mcp_tools: List[Any]):
        """Carga y organiza las herramientas por tags (Misma lógica que tenías)."""
        self.tools = []
        self._tools_by_tag = {}
        for tool in mcp_tools:
            tags = set()
            # Extraer tags de la metadata de FastMCP
            if hasattr(tool, "meta") and isinstance(tool.meta, dict):
                fastmcp_meta = tool.meta.get("_fastmcp", {})
                if isinstance(fastmcp_meta, dict):
                    tags.update(fastmcp_meta.get("tags", []))

            # Formato de herramienta para Ollama (OpenAI compatible)
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": self._sanitize_schema(tool.inputSchema),
                },
                "tags": list(tags),
            }
            self.tools.append(tool_dict)
            for tag in tags:
                if tag not in self._tools_by_tag: 
                    self._tools_by_tag[tag] = []
                self._tools_by_tag[tag].append(tool_dict)

    def _sanitize_schema(self, schema: Any) -> Dict[str, Any]:
        """Limpia el schema asegurando que las descripciones de parámetros se mantengan."""
        if hasattr(schema, "model_dump"): 
            schema = schema.model_dump()
            
        if isinstance(schema, dict):
            # Mantenemos 'description' si existe a nivel de parámetro
            # Eliminamos campos técnicos de JSON Schema que confunden a modelos locales (Ollama)
            forbidden_keys = {"additionalProperties", "$schema", "definitions"}
            new_schema = {k: v for k, v in schema.items() if k not in forbidden_keys}
            
            for key, value in new_schema.items():
                if isinstance(value, dict):
                    new_schema[key] = self._sanitize_schema(value)
            return new_schema
        return schema

    def set_active_tools(self, include_tags: List[str]):
        """Filtra las herramientas por tags (Misma lógica que tenías)."""
        filtered = []
        seen = set()
        for tag in include_tags:
            if tag in self._tools_by_tag:
                for t in self._tools_by_tag[tag]:
                    if t["function"]["name"] not in seen:
                        filtered.append(t)
                        seen.add(t["function"]["name"])
        self.active_tools = filtered

    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str
    ) -> Dict[str, Any]:
        # 1. Preparar historial
        history = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = "assistant" if m["role"] in ["model", "assistant"] else "user"
            history.append({"role": role, "content": m["content"]})

        # 2. Llamada a Ollama
        response = await self.client.chat(
            model=self.model_name,
            messages=history,
            tools=self.tools if self.tools else None,
            options={"temperature": 0.7} # Bajamos temperatura para evitar alucinaciones
        )

        print(response)

        message = response.get("message", {})
        
        # CASO A: Ollama detecta la herramienta nativamente (Lo ideal)
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]["function"]
            return {
                "tool": tool_call["name"],
                "params": tool_call["arguments"]
            }
        
        
        # # CASO B: Qwen escribe el JSON en el 'content' (Lo que te ha pasado)
        # content = message.get("content", "").strip()
        # if content:
        #     # Buscamos si hay un JSON de herramienta dentro del texto
        #     import json
        #     import re
            
        #     # Intentamos buscar patrones de JSON de herramientas
        #     try:
        #         # Si el contenido es un JSON puro
        #         if content.startswith("{"):
        #             data = json.loads(content)
        #             if "name" in data:
        #                 return {
        #                     "tool": data["name"],
        #                     "params": data.get("arguments", data.get("params", {}))
        #                 }
                
        #         # Si el JSON está envuelto en texto o bloques de código ```json
        #         json_match = re.search(r'\{.*"name".*\}', content, re.DOTALL)
        #         if json_match:
        #             data = json.loads(json_match.group())
        #             return {
        #                 "tool": data["name"],
        #                 "params": data.get("arguments", data.get("params", {}))
        #             }
        #     except Exception as e:
        #         logger.debug(f"No se pudo parsear JSON del contenido: {e}")

        # # CASO C: Respuesta de texto normal
        # return {"content": content, "tool": None}