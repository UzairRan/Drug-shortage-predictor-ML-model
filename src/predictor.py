import joblib
import numpy as np
import json
from tensorflow.keras.models import load_model

class DrugShortagePredictor:
    def __init__(self):
        # Load FDA classification model
        self.fda_model = joblib.load("models/fda_classifier.pkl")

        # ✅ Load threshold directly (assuming JSON file contains a float like 0.76)
        with open("models/fda_optimal_threshold.json", "r") as f:
            self.fda_threshold = json.load(f)["optimal_threshold"]


        # Load CMS LSTM model (without compiling to avoid TensorFlow version issues)
        self.cms_model = load_model("models/cms_lstm_model.h5", compile=False)

        # Load scalers
        self.scaler_X = joblib.load("models/scaler_X.save")
        self.scaler_y = joblib.load("models/scaler_y.save")

    def predict_fda(self, features_array):
        proba = self.fda_model.predict_proba([features_array])[0][1]
        pred = int(proba >= self.fda_threshold)
        return {
            "Shortage_Risk": bool(pred),
            "Probability": {
                "No": round(float(1 - proba), 2),
                "Yes": round(float(proba), 2)
            }
        }

    def predict_cms_utilization(self, sequence_array):
        sequence_array = np.array(sequence_array).reshape(1, *sequence_array.shape)
        pred_scaled = self.cms_model.predict(sequence_array)[0][0]
        utilization = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
        return {"Predicted_Utilization": round(float(utilization))}
