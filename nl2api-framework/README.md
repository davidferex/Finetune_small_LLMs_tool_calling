# NL2API Framework 🚀
Un framework para generar interfaces de lenguaje natural para cualquier API mediante fine tuning de modelos locales eficientes (Gemma 2).

## 💡 Concepto
Este framework permite que cualquier desarrollador pase de una definición de API estática en formato JSON a una interfaz conversacional capaz de:
1. **Mapear intención a código:** Traducir lenguaje natural a llamadas JSON.
2. **Gestionar ambigüedad:** Detectar parámetros faltantes de forma automática.
3. **Ejecutar en local:** Optimizado para correr en Gemma 2 2B.

## 🛠️ Cómo funciona
1. **Define tu API:** Crea un `tools_spec.json` con tus funciones y parámetros.
2. **Genera el Dataset:** El `DatasetGenerator` usa un modelo maestro (Llama 3.3) para crear miles de interacciones variadas.
3. **Fine-tune:** Entrena un modelo pequeño para que aprenda *tu* API específica.
4. **Despliega:** Usa el `InferenceBridge` para recibir texto y ejecutar funciones reales.

## 🧪 Caso de Uso: Simulador de Datos Sintéticos
Hemos validado el framework creando una interfaz para una API de generación de datos: **[CalmDataGenerator](https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator)**
- **Input:** "Necesito 100 filas de una serie temporal rápido."
- **Output:** `TimeSeriesPreset(n_samples=100, ...)`
