from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return {"message": "Customer Segmentation API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["age"]
    income = data["income"]
    spending = data["spending"]

    X = np.array([[age, income, spending]])
    X_scaled = scaler.transform(X)

    cluster = model.predict(X_scaled)

    return jsonify({
        "cluster": int(cluster[0])
    })

if __name__ == "__main__":
    app.run()
