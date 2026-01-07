class BasePrompt:
    def __init__(self, stats=None):
        self.base_template = "Eres un experto en bioinformática encargado de ejecutar un pipeline de Data Wrangling. Responde SOLO con la tool que elijas para la situación planteada. No incluyas ninguno de tus razonamientos o pensamientos, SOLO el nombre de la tool."
        self.specific_instruction = ""
        # Aseguramos que stats sea un dict para que .get() no falle nunca
        self.stats = stats if isinstance(stats, dict) else {}

    def get_content(self, research_question=""):
        # Extraemos las claves EXACTAS que devuelve tu función Get_data_statistics
        genes = self.stats.get('current_genes', '0')
        v_mean = self.stats.get('mean_var', 'N/A')
        v_max = self.stats.get('max_var', 'N/A')
        status = self.stats.get('status', 'Unknown')

        info_data = (
            f"\n\n--- ESTADO DEL DATASET ---\n"
            f"- Genes actuales: {genes}\n"
            f"- Varianza Media: {v_mean}\n"
            f"- Varianza Máxima: {v_max}\n"
            f"- Estatus: {status}"
        )
        
        # Combinamos: Madre + Hija + Pregunta + Stats
        return (
            f"CONTEXTO: {self.base_template}\n"
            f"FASE ACTUAL: {self.specific_instruction}\n"
            f"DATASET: {self.stats}\n"
            f"OBJETIVO: {research_question}\n\n"
            "REGLAS ESTRICTAS:\n"
            "1. PROHIBIDO RESPONDER CON TEXTO HUMANO.\n"
            "2. ELIGE UNA HERRAMIENTA DE LA LISTA.\n"
            "3. DEVUELVE EL NOMBRE EXACTO DE LA HERRAMIENTA ESCOGIDA."
        )


class FilterPrompt(BasePrompt):
    def __init__(self, stats=None):
        super().__init__(stats)
        self.specific_instruction = "FASE: Filtrado."

class NormalizePrompt(BasePrompt):
    def __init__(self, stats=None):
        super().__init__(stats)
        self.specific_instruction = "FASE: Normalización."