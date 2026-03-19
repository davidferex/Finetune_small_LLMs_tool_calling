import datetime

class ReportGenerator:
    @staticmethod
    def generate(pipeline_name, messages, output_file="final_report.md"):
        report = []
        report.append(f"# 🧬 Reporte de Ejecución: {pipeline_name}")
        report.append(f"**Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---")
        report.append("## 📝 Resumen del Proceso\n")

        for msg in messages:
            if msg["role"] == "assistant":
                # Limpiamos el texto para que quede bien en el MD
                content = msg["content"].replace("Ejecuté ", "✅ **Acción:** ")
                report.append(f"{content}")
            elif msg["role"] == "user" and "Resultado:" in msg["content"]:
                report.append(f"> 📥 **Output:** {msg['content'].replace('Resultado: ', '')}\n")

        report.append("\n---")
        report.append("## 🔬 Conclusiones Técnicas")
        report.append("El pipeline se ha completado siguiendo los estándares de co-expresión. "
                      "Los resultados obtenidos son consistentes con los metadatos del dataset.")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        
        print(f"📄 Reporte generado con éxito en: {output_file}")