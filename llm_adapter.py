import os
import logging
import asyncio
from google import genai
from google.genai import types
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class LLMAdapter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash"
    ):
        self.model_name = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # Inicializamos el cliente de la nueva librería
        self.client = genai.Client(api_key=self.api_key)
        
        self.tools = []
        self._tools_by_tag = {}
        self.active_tools = []

    def load_tools(self, mcp_tools: List[Any]):
        """Carga y organiza las herramientas por tags."""
        self.tools = []
        self._tools_by_tag = {}
        for tool in mcp_tools:
            tags = set()
            if hasattr(tool, "meta") and isinstance(tool.meta, dict):
                fastmcp_meta = tool.meta.get("_fastmcp", {})
                if isinstance(fastmcp_meta, dict):
                    tags.update(fastmcp_meta.get("tags", []))

            tool_dict = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": self._sanitize_schema_for_gemini(tool.inputSchema),
                "tags": list(tags),
            }
            self.tools.append(tool_dict)
            for tag in tags:
                if tag not in self._tools_by_tag: self._tools_by_tag[tag] = []
                self._tools_by_tag[tag].append(tool_dict)

    def _sanitize_schema_for_gemini(self, schema: Any) -> Dict[str, Any]:
        """Lógica de limpieza de JSON Schema necesaria para Gemini."""
        if hasattr(schema, "model_dump"): schema = schema.model_dump()
        if isinstance(schema, dict):
            schema = {k: v for k, v in schema.items() if k != "additionalProperties"}
            for key, value in schema.items():
                if isinstance(value, dict):
                    schema[key] = self._sanitize_schema_for_gemini(value)
        return schema

    def set_active_tools(self, include_tags: List[str]):
        """Filtra las herramientas que se enviarán en la próxima llamada."""
        filtered = []
        seen = set()
        for tag in include_tags:
            if tag in self._tools_by_tag:
                for t in self._tools_by_tag[tag]:
                    if t["name"] not in seen:
                        filtered.append(t); seen.add(t["name"])
        self.active_tools = filtered

    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str
    ) -> Dict[str, Any]:
        """
        Realiza la llamada configurando el System Prompt dinámicamente con la nueva SDK.
        """
        # 1. Convertir herramientas activas a Function Declarations usando types de la nueva SDK
        function_declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"] # La nueva SDK acepta el dict directo si está saneado
            ) for t in self.active_tools
        ]

        # 2. Configurar el objeto Tool y el ToolConfig (Modo ANY para forzar uso de herramientas)
        tools = [types.Tool(function_declarations=function_declarations)] if function_declarations else None
        
        tool_config = None
        if function_declarations:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY" 
                )
            )

        # 3. Configuración de la generación
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools,
            tool_config=tool_config
        )

        # 4. Preparar el mensaje (La nueva SDK prefiere el mensaje actual)
        last_msg = messages[-1]["content"]

        # 5. Llamada asíncrona usando el cliente nuevo
        # Nota: La nueva SDK es síncrona por defecto, usamos to_thread para no bloquear
        response = await asyncio.to_thread(
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=last_msg,
                config=config
            )
        )

        # 6. Retornar la herramienta elegida
        # La estructura de respuesta ha cambiado ligeramente en la nueva SDK
        part = response.candidates[0].content.parts[0]
        
        if part.function_call:
            return {
                "tool": part.function_call.name,
                "params": part.function_call.args # Ya viene como dict
            }
        return {"content": part.text if hasattr(part, "text") else "", "tool": None}

    # Nota: El mapeo manual a genai.protos.Schema ya no es estrictamente necesario 
    # con la nueva SDK si el dict está limpio, pero lo he integrado en la lógica de arriba.