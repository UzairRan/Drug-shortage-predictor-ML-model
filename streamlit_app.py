import streamlit as st
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

# Load models and scalers
fda_model = joblib.load("models/fda_classifier.pkl")
with open("models/fda_optimal_threshold.json", "r") as f:
    fda_threshold = json.load(f)["optimal_threshold"]
cms_model = load_model("models/cms_lstm_model.h5", compile=False)
scaler_X = joblib.load("models/scaler_X.save")
scaler_y = joblib.load("models/scaler_y.save")

st.set_page_config(page_title="Drug Shortage Predictor", layout="centered")
st.title("💊 Drug Shortage Prediction System")
st.markdown("Predict drug shortage risk or utilization trends using trained ML/DL models.")

# ---- FDA CLASSIFICATION ----
st.header("📍 FDA Shortage Classifier")
st.markdown("Enter FDA features as comma-separated values (Total 11 features required):")
fda_input = st.text_input("Example: 0,12,3,0,0,0,0,0,0,0,0")

if st.button("Predict Shortage (FDA)"):
    try:
        features = list(map(float, fda_input.strip().split(",")))
        if len(features) != 11:
            st.error(f"Feature shape mismatch, expected: 11, got {len(features)}")
        else:
            proba = fda_model.predict_proba([features])[0][1]
            pred = int(proba >= fda_threshold)
            st.success("Prediction Complete")
            st.json({
                "Shortage Risk": bool(pred),
                "Probability": {
                    "No": round(1 - proba, 2),
                    "Yes": round(proba, 2)
                }
            })
    except Exception as e:
        st.error(f"Error: {e}")

# ---- CMS LSTM FORECAST ----
st.header("📈 CMS Utilization Forecast")
st.markdown("Input LSTM-ready feature sequence (shape: 3x5)")
st.markdown("Format: [[feat1,feat2,...], [feat1,...], [feat1,...]]")
cms_input = st.text_area("Enter sequence (3 timesteps, 5 features each):", "[[2021, 1, 2, 1, 1345], [2022, 1, 2, 1, 1452], [2023, 1, 2, 1, 1570]]")

if st.button("Forecast Utilization (CMS)"):
    try:
        sequence = np.array(eval(cms_input))
        sequence_scaled = scaler_X.transform(sequence)
        sequence_scaled = sequence_scaled.reshape(1, *sequence_scaled.shape)
        pred_scaled = cms_model.predict(sequence_scaled)[0][0]
        utilization = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        st.success("Forecast Complete")
        st.metric("Predicted Utilization", f"{utilization:.0f}")
    except Exception as e:
        st.error(f"Error: {e}")
