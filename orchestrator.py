import asyncio
import ast
import time
from mcp import ClientSession

class BioOrchestrator:
    def __init__(self, session: ClientSession, adapter, pipeline_config):
        self.session = session
        self.adapter = adapter
        self.config = pipeline_config
        self.current_state = pipeline_config["initial_state"]
        self.messages = []
        self.pipeline_active = True

        # --- NUEVOS ATRIBUTOS DE CONTROL ---
        self.last_call_time = 0
        self.min_interval = 13  # 13 segundos para estar seguros con el margen de 5 RPM
        self.total_tokens_estimated = 0
    
    async def _wait_for_rate_limit(self):
        """Asegura que respetamos el límite de 5 llamadas por minuto (RPM)."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"⏳ [Rate Limit] Esperando {wait_time:.1f}s para la próxima llamada...")
            await asyncio.sleep(wait_time)
        self.last_call_time = time.time()

    async def run(self, research_question: str):
        print(f"\n🚀 Starting Pipeline: {self.config['name']}")
        
        # Initialize context with the main objective
        self.messages.append({
            "role": "user", 
            "content": f"Objective: {research_question}"
        })

        while self.pipeline_active:
            state_info = self.config["states"][self.current_state]
            
            # 1. Update context with current Step Stats
            stats = await self._get_stats()

            # 2. Get dynamic prompt for the current state
            prompt_class = state_info["prompt_class"]
            prompt_obj = prompt_class(stats=stats)
            # Removed research_question from get_content to match your new prompts.py
            system_prompt = prompt_obj.get_content() 

            # 3. Restrict tools to current state + the 'move' tool
            active_tools = state_info["tools"] + ["MOVE"]
            self.adapter.set_active_tools(active_tools)

            print(f"\n🧠 [Current State: {self.current_state}] Reasoning...")

            await self._wait_for_rate_limit()

            # 4. Chat with LLM (Iteration within the same state)
            resultado = await self.adapter.chat(
                messages=self.messages,
                system_prompt=system_prompt
            )

            if resultado["tool"]:
                # 5. Tool Execution
                print(f"🛠️  Executing: {resultado['tool']} with {resultado['params']}")
                tool_output = await self.session.call_tool(resultado["tool"], resultado["params"])
                observacion = tool_output.content[0].text
                
                # Update Chat History
                self.messages.append({"role": "assistant", "content": f"Action: Called {resultado['tool']}"})
                self.messages.append({"role": "user", "content": f"Observation: {observacion}"})

                # 6. Logic for State Transition (The "Enrouter")
                if resultado["tool"] == "move_to_next_step":
                    # Logic-based validation: Verify if we can actually move
                    if self._verify_transition_readiness(self.current_state, observacion):
                        print(f"✅ State Transition Approved: Moving from {self.current_state} to {state_info['next']}")
                        self.current_state = state_info["next"]
                        
                        if self.current_state == "DONE":
                            self.pipeline_active = False
                    else:
                        # Feedback loop if transition is premature
                        self.messages.append({
                            "role": "user", 
                            "content": "Transition failed. Requirements for this step are not fully met. Please verify your results."
                        })
                
                # Note: If it wasn't 'move_to_next_step', the loop continues in the SAME state.
            else:
                # If LLM returns content without a tool call
                print(f"💬 Assistant says: {resultado['content']}")
                # We stop to avoid infinite loops if it doesn't call a tool
                if not resultado["tool"] and self.current_state != "DONE":
                     print("⚠️ LLM stopped without calling move_to_next_step.")
                     self.pipeline_active = False
            
            if not self.pipeline_active:
                print("\n🏁 Pipeline finished.")
                from report_generator import ReportGenerator
                ReportGenerator.generate(self.config["name"], self.messages)

    def _verify_transition_readiness(self, current_state, summary):
        """
        Optional: Hybrid Router logic. 
        You can check if specific files exist or if the 'summary' 
        contains keywords.
        """
        # For your TFM, we can start with simple True, or add checks:
        # if current_state == "NETWORK" and "beta" not in summary.lower(): return False
        return True

    async def _get_stats(self):
        try:
            output = await self.session.call_tool("get_data_statistics", {})
            content = output.content[0].text
            return ast.literal_eval(content) if isinstance(content, str) else content
        except Exception as e:
            print(f"⚠️ Stats fetch failed: {e}")
            return {}