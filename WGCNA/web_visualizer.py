from flask import Flask, render_template_string, request, jsonify
from datetime import datetime

app = Flask(__name__)

# ---- Global store ----
results_store = {
    "status": "Waiting for data...",
    "data": None,
    "history": [],
    "llm_summary": None,
    "notenotebook_blocks": None
}


# ---- RECEIVE RESULTS ----
@app.route('/update_results', methods=['POST'])
def update_results():
    global results_store

    data = request.json
    results_store["data"] = data
    results_store["status"] = "Analysis Received"

    # ---- Save history snapshot ----
    snapshot = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "modules": data.get("summary", {}).get("n_modules"),
        "genes": data.get("summary", {}).get("total_genes"),
        "beta": data.get("summary", {}).get("beta")
    }

    results_store["history"].insert(0, snapshot)
    results_store["history"] = results_store["history"][:10]

    return jsonify({"status": "success"})


# ---- NEW: RECEIVE LLM SUMMARY ----
@app.route('/update_summary', methods=['POST'])
def update_summary():
    global results_store
    data = request.json

    results_store["llm_summary"] = data.get("summary_text")

    return jsonify({"status": "success"})


# ---- NEW: RECEIVE NOTEBOOK ----
@app.route('/update_notebook_blocks', methods=['POST'])
def update_notebook_blocks():
    global results_store

    data = request.json

    results_store["notebook_blocks"] = {
        "block_1": data.get("block_1"),
        "block_2": data.get("block_2")
    }

    return jsonify({"status": "notebook blocks updated"})


# ---- MAIN PAGE ----
@app.route('/')
def index():

    data = results_store
    heatmap = data["data"].get("heatmap") if data["data"] else None
    top_module = data["data"].get("top_module") if data["data"] else None
    plots = data["data"].get("plots") if data["data"] else None
    history = data.get("history", [])
    llm_summary = data.get("llm_summary")
    notebook = data.get("notebook_blocks")   # ✅ añadido

    return render_template_string("""
<html>
<head>
<title>WGCNA Live Dashboard</title>

<style>
body {
    font-family: 'Segoe UI', sans-serif;
    padding: 40px;
    background: #f6f8fb;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

h1 { color: #1a73e8; }

.stat-box {
    display: inline-block;
    padding: 10px 20px;
    background: #e8f0fe;
    border-radius: 8px;
    margin-right: 10px;
    margin-bottom: 15px;
}

img {
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 8px;
    text-align: center;
}

th {
    background: #f1f3f4;
}

.history-row {
    font-size: 14px;
    padding: 6px 0;
    border-bottom: 1px solid #eee;
}

.summary-box {
    background: #f1f3f4;
    padding: 18px;
    border-radius: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
}

/* ✅ NOTEBOOK STYLE AÑADIDO */
.nb-cell {
    margin-bottom: 20px;
}

.nb-title {
    font-weight: 600;
    margin-bottom: 6px;
    color: #444;
}

.nb-code {
    background: #1e1e1e;
    color: #e6e6e6;
    padding: 16px;
    border-radius: 12px;
    font-family: monospace;
    font-size: 14px;
    white-space: pre-wrap;
    overflow-x: auto;
}
</style>

<script>
setInterval(async () => {
    const res = await fetch("/");
    const text = await res.text();
    document.open();
    document.write(text);
    document.close();
}, 15000);
</script>

</head>

<body>

<div class="card">
<h1>📊 WGCNA Dashboard — {{ data.status }}</h1>

{% if data.data %}
<div class="stat-box">Modules: {{ data.data.summary.n_modules }}</div>
<div class="stat-box">Grey genes: {{ data.data.summary.grey_genes }}</div>
<div class="stat-box">Total genes: {{ data.data.summary.total_genes }}</div>
<div class="stat-box">Beta: {{ data.data.summary.beta }}</div>
{% endif %}
</div>


{% if plots and plots.dendrogram %}
<div class="card">
<h2>🌳 Gene Dendrogram</h2>
<img src="data:image/png;base64,{{ plots.dendrogram }}" style="width:100%;">
</div>
{% endif %}


{% if plots and plots.heatmap %}
<div class="card">
<h2>🔥 Module–Trait Heatmap (Visual)</h2>
<img src="data:image/png;base64,{{ plots.heatmap }}" style="width:80%;">
</div>
{% endif %}


{% if heatmap %}
<div class="card">
<h2>📈 Module–Trait Correlations (Numeric)</h2>

<table border="1">
<tr>
<th>Module</th>
{% for trait in heatmap.traits %}
<th>{{ trait }} (r)</th>
<th>{{ trait }} (p)</th>
{% endfor %}
</tr>

{% for i in range(heatmap.modules|length) %}
<tr>
<td><b>ME_{{ heatmap.modules[i] }}</b></td>

{% for j in range(heatmap.traits|length) %}
{% set r_val = heatmap.r_matrix[i][j] %}
{% set p_val = heatmap.p_matrix[i][j] %}

<td style="color: {{ 'red' if r_val is not none and r_val|abs > 0.7 else 'black' }}; font-weight: bold;">
{{ r_val|round(4) if r_val is not none else 'NA' }}
</td>

<td>
{{ "%.2e"|format(p_val) if p_val is not none else 'NA' }}
</td>

{% endfor %}
</tr>
{% endfor %}
</table>

</div>
{% endif %}


{% if top_module %}
<div class="card">
<h2>🏆 Top Module</h2>
<p>
Module <b>ME_{{ top_module.module }}</b><br>
Trait: {{ top_module.trait }}<br>
r = {{ top_module.r|round(4) }} |
p = {{ "%.2e"|format(top_module.p) }}
</p>
</div>
{% endif %}


{% if llm_summary %}
<div class="card">
<h2>🧠 LLM Analysis Summary</h2>
<div class="summary-box">
{{ llm_summary }}
</div>
</div>
{% endif %}


<!-- ✅ NOTEBOOK RENDER AÑADIDO -->
{% if notebook %}
<div class="card">
<h2>📓 Notebook Execution</h2>

{% if notebook.block_1 %}
<div class="nb-cell">
<div class="nb-title">Cell 1</div>
<div class="nb-code">{{ notebook.block_1 }}</div>
</div>
{% endif %}

{% if notebook.block_2 %}
<div class="nb-cell">
<div class="nb-title">Cell 2</div>
<div class="nb-code">{{ notebook.block_2 }}</div>
</div>
{% endif %}

</div>
{% endif %}


{% if history %}
<div class="card">
<h2>🕓 Run History</h2>

{% for h in history %}
<div class="history-row">
{{ h.time }} → Modules: {{ h.modules }} | Genes: {{ h.genes }} | Beta: {{ h.beta }}
</div>
{% endfor %}

</div>
{% endif %}


{% if not data.data %}
<div class="card">
Pipeline running... Waiting for expose_results_to_web()
</div>
{% endif %}


</body>
</html>
""", data=data, heatmap=heatmap, top_module=top_module, plots=plots, history=history, llm_summary=llm_summary, notebook=notebook)  # ✅ añadido


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
