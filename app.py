from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load("kmeans_model.pkl")
escalador = joblib.load("scaler.pkl")

CLUSTER_LABELS = {
    0: "Bajo ingreso – Bajo gasto (Clientes cuidadosos)",
    1: "Bajo ingreso – Alto gasto (Clientes impulsivos)",
    2: "Alto ingreso – Alto gasto (Clientes premium)",
    3: "Alto ingreso – Bajo gasto (Clientes conservadores)",
    4: "Ingreso medio – Gasto medio (Clientes estándar)"
}

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Segmentación de Clientes</title>
    <style>
        body { font-family: Arial; background: #f4f6f8; }
        .box {
            width: 420px;
            margin: 80px auto;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        input, button {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
        }
        button {
            background: #2563eb;
            color: white;
            border: none;
            cursor: pointer;
        }
        h2 { text-align: center; }
        .result {
            margin-top: 15px;
            padding: 10px;
            background: #e0f2fe;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h2>Segmentación de Clientes</h2>
        <form method="post">
            <input type="number" name="age" placeholder="Edad" required>
            <input type="number" name="income" placeholder="Ingreso anual (miles $)" required>
            <input type="number" name="spending" placeholder="Nivel de gasto (1-100)" required>
            <button type="submit">Calcular segmento</button>
        </form>

        {% if cluster is not none %}
        <div class="result">
            <strong>Cluster {{ cluster }}</strong><br>
            {{ label }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def inicio():
    cluster = None
    label = None

    if request.method == "POST":
        edad = float(request.form["age"])
        ingreso = float(request.form["income"])
        gasto = float(request.form["spending"])

        X = np.array([[edad, ingreso, gasto]])
        X_scaled = escalador.transform(X)
        cluster = int(modelo.predict(X_scaled)[0])
        label = CLUSTER_LABELS.get(cluster, "Segmento desconocido")

    return render_template_string(HTML_PAGE, cluster=cluster, label=label)

@app.route("/predict", methods=["POST"])
def predecir():
    data = request.get_json()

    X = np.array([[data["age"], data["income"], data["spending"]]])
    X_scaled = escalador.transform(X)
    cluster = int(modelo.predict(X_scaled)[0])

    return jsonify({
        "cluster": cluster,
        "segmento": CLUSTER_LABELS.get(cluster, "Segmento desconocido")
    })

if __name__ == "__main__":
    app.run()
