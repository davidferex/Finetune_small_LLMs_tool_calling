import asyncio
import ast
from mcp import ClientSession

class BioOrchestrator:
    def __init__(self, session: ClientSession, adapter, pipeline_config):
        self.session = session
        self.adapter = adapter
        self.config = pipeline_config
        self.current_state = pipeline_config["initial_state"]
        self.messages = []
        self.pipeline_active = True

    async def run(self, research_question: str):
        print(f"\n🚀 Iniciando Pipeline: {self.config['name']}")
        
        # Inicializamos el contexto
        self.messages.append({
            "role": "user", 
            "content": f"Peticion: {research_question}. Por favor, aplica el Data Wrangling necesario."
        })

        pasos = 0
        while self.pipeline_active:
            pasos += 1
            state_info = self.config["states"][self.current_state]

            # 1. Obtención de estadísticas (Feedback Loop)
            stats = await self._get_stats()

            # 2. Preparar el Prompt dinámico
            # Instanciamos la clase de prompt definida en el JSON
            prompt_class = state_info["prompt_class"]
            prompt_obj = prompt_class(stats=stats)
            system_prompt = prompt_obj.get_content(research_question)

            # 3. Restringir herramientas al estado actual
            self.adapter.set_active_tools(state_info["tools"])

            print(f"\n🧠 [Paso {pasos} - Estado: {self.current_state}] Razonando...")

            # 4. Llamada al LLM
            resultado = await self.adapter.chat(
                messages=self.messages,
                system_prompt=system_prompt
            )

            if resultado["tool"]:
                # 5. Acción y Observación
                print(f"🛠️  Ejecutando: {resultado['tool']} con {resultado['params']}")
                tool_output = await self.session.call_tool(resultado["tool"], resultado["params"])
                observacion = tool_output.content[0].text
                
                # Actualizar memoria
                self.messages.append({"role": "assistant", "content": f"Ejecuté {resultado['tool']}"})
                self.messages.append({"role": "user", "content": f"Resultado: {observacion}"})

                # 6. Transición de estado
                self.current_state = state_info["next"]
                if self.current_state == "DONE":
                    self.pipeline_active = False
            else:
                print(f"🏁 Finalizado: {resultado['content']}")
                self.pipeline_active = False

    async def _get_stats(self):
        output = await self.session.call_tool("Get_data_statistics", {})
        content = output.content[0].text
        try:
            return ast.literal_eval(content) if isinstance(content, str) else content
        except:
            return {}