import json
import importlib

class PipelineLoader:
    @staticmethod
    def load(json_path, module_name="prompts"):
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Importamos el módulo donde están las clases de los prompts
        module = importlib.import_module(module_name)
        
        for state_name, state_data in config["states"].items():
            class_name = state_data["prompt_class"]
            
            # Obtenemos la clase real a partir del string
            prompt_class = getattr(module, class_name)
            
            # Reemplazamos el string por la clase (objeto)
            state_data["prompt_class"] = prompt_class
            
        return config