from flask import Flask, request, jsonify
from src.predictor import DrugShortagePredictor
import numpy as np

app = Flask(__name__)
predictor = DrugShortagePredictor()

@app.route('/')
def home():
    return "💊 Drug Shortage Prediction API is running..."

@app.route('/predict/fda', methods=['GET', 'POST'])
def predict_fda():
    data = request.json
    features = data.get("features", None)  # This should be your input array for FDA model
    if features is None:
        return jsonify({"error": "Missing 'features'"}), 400
    result = predictor.predict_fda(features)
    return jsonify(result)

@app.route('/predict/cms', methods=['POST'])
def predict_cms():
    data = request.json
    sequence = data.get("sequence", None)  # This should be 3D sequence for LSTM
    if sequence is None:
        return jsonify({"error": "Missing 'sequence'"}), 400
    result = predictor.predict_cms_utilization(np.array(sequence))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)



