import pandas as pd
import numpy as np
import json
from fastmcp import FastMCP

mcp = FastMCP(name="GeneProcessingMCP")

pipeline_state = {
    "data_loaded": False,
    "filtered_top_n": False,
    "filtered_var_threshold": False,
    "normalized_quantiles": False,
    "normalized_zscore": False,
    "pipeline_complete": False
}

file_path = "./data_genes.json"

# --- ESTADO GLOBAL ---
df_global = None

# --- HERRAMIENTAS DE FILTRADO ---

@mcp.tool(tags=["FILTER"], description="Mantiene solo los N genes con mayor expresión media. Útil para reducción de dimensionalidad enfocada en señales fuertes.")
def Filter_genes_top_n(n: int = 100):
    global df_global
    if df_global is None: return "Error: Sin datos."
    
    before = len(df_global)
    mean_exp = df_global.mean(axis=1)
    df_global = df_global.loc[mean_exp.nlargest(n).index]
    
    pipeline_state["filtered_top_n"] = True
    return f"Filtrado Top N completado. De {before} a {len(df_global)} genes."

@mcp.tool(tags=["FILTER"], description="Elimina genes con baja variabilidad según el umbral V. Ideal para limpieza de ruido y eliminación de genes no informativos.")
def Filter_genes_var_threshold(v: float = 0.5):
    global df_global
    if df_global is None: return "Error: Sin datos."
    
    before = len(df_global)
    variances = df_global.var(axis=1)
    df_global = df_global[variances > v]
    
    pipeline_state["filtered_var_threshold"] = True
    return f"Filtrado varianza aplicado. De {before} a {len(df_global)} genes."

# --- HERRAMIENTAS DE NORMALIZACIÓN ---

@mcp.tool(tags=["NORM"], description="Aplica normalización por cuantiles para homogeneizar distribuciones entre muestras. Recomendado para corregir variaciones técnicas.")
def Normalize_quantiles():
    global df_global
    if df_global is None: return "Error: Sin datos."
    
    # Lógica de Quantile Normalization
    rank_mean = df_global.stack().groupby(df_global.rank(method='first').stack().astype(int)).mean()
    df_global = df_global.rank(method='first').stack().astype(int).map(rank_mean).unstack()
    
    pipeline_state["normalized_quantiles"] = True
    pipeline_state["pipeline_complete"] = True
    return "Normalización por cuantiles completada."

@mcp.tool(tags=["NORM"], description="Escala los datos a Z-score (media 0, std 1) por gen. Necesario para algoritmos que requieren datos estandarizados.")
def Normalize_zscore():
    global df_global
    if df_global is None: return "Error: Sin datos."
    
    df_global = df_global.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x, axis=1)
    
    pipeline_state["normalized_zscore"] = True
    pipeline_state["pipeline_complete"] = True
    return "Normalización Z-score completada."

# --- HERRAMIENTAS DE INFORMACIÓN ---

@mcp.tool(tags=["INFO"], description="Proporciona estadísticas técnicas del dataset actual (varianza, conteo de genes) para decidir umbrales de filtrado.")
def Get_data_statistics():
    global df_global
    if df_global is None: 
        try:
            # Cargamos asumiendo orientación de índice (Gene_001: {S1: val, S2: val})
            df_global = pd.read_json(file_path, orient="index")
            pipeline_state["data_loaded"] = True
            print("JSON cargado en DataFrame con éxito")
        except Exception as e:
            print("Error cargando el JSON. Mensaje de error: " + str(e))
    variances = df_global.var(axis=1)
    return {
        "max_var": float(variances.max()),
        "min_var": float(variances.min()),
        "mean_var": float(variances.mean()),
        "current_genes": len(df_global)
    }

@mcp.tool(tags=["INFO"], description="Devuelve el estado de ejecución de todos los pasos del pipeline de Data Wrangling.")
def get_pipeline_state():
    return {"success": True, "state": pipeline_state}