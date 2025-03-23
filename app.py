import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load the trained model
logging.info("Loading trained model...")
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]  # Get input data
        prediction = model.predict([data]).tolist()  # Make prediction
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
