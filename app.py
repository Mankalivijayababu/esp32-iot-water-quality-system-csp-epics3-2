from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import math
import os
import time
import json
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # allow frontend calls

# ============================================================
# HELPERS
# ============================================================
def clean(value):
    try:
        if value is None:
            return 0
        if isinstance(value, float) and math.isnan(value):
            return 0
        return float(value)
    except:
        return 0


def get_float(data, *keys, default=0.0):
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except:
                pass
    return float(default)


# ============================================================
# LOAD MODELS
# ============================================================
print("Loading RandomForest model...")
with open("rf_model.pkl", "rb") as f:
    classifier_model = pickle.load(f)

print("Loading LabelEncoder...")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Loading LSTM model...")
lstm_model = load_model("lstm_model.h5")
lstm_scaler = joblib.load("scaler.pkl")

df = pd.read_csv("water_quality_big_dataset.csv", parse_dates=["Date"]).sort_values("Date")
df = df.reset_index(drop=True)

print("Backend Ready!")


# ============================================================
# 1) Random Forest
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        tds = get_float(data, "TDS", "tds", default=0.0)
        turb = get_float(data, "Turbidity", "turbidity", default=0.0)

        X = pd.DataFrame([[tds, turb]], columns=["TDS", "Turbidity"])
        pred_class = classifier_model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        probabilities = classifier_model.predict_proba(X)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 2) LSTM FUTURE FORECAST (TDS + Turbidity)
# ============================================================
@app.route("/predict_future", methods=["POST", "GET"])
def predict_future():
    try:
        if request.method == "POST":
            data = request.get_json()
            steps = int(data.get("steps", 7))
        else:
            steps = int(request.args.get("steps", 7))

        window = 14
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled = lstm_scaler.transform(recent)
        seq = np.array([scaled])

        predictions = []

        for _ in range(steps):
            pred_scaled = lstm_model.predict(seq, verbose=0)[0]
            real = lstm_scaler.inverse_transform([pred_scaled])[0]

            predictions.append({
                "TDS": float(real[0]),
                "Turbidity": float(real[1])
            })

            seq = np.array([np.vstack([seq[0][1:], pred_scaled])])

        return jsonify({"future_predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3) LSTM QUALITY FORECAST
# ============================================================
@app.route("/predict_future_quality", methods=["POST"])
def predict_future_quality():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 7))

        window = 14
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled = lstm_scaler.transform(recent)
        seq = np.array([scaled])

        results = []

        for _ in range(steps):
            pred_scaled = lstm_model.predict(seq, verbose=0)[0]
            real = lstm_scaler.inverse_transform([pred_scaled])[0]

            tds = float(real[0])
            turb = float(real[1])

            if tds < 500:
                q = "Safe"
            elif tds < 1000:
                q = "Moderate"
            else:
                q = "Unsafe"

            results.append({
                "TDS": tds,
                "Turbidity": turb,
                "Quality": q
            })

            seq = np.array([np.vstack([seq[0][1:], pred_scaled])])

        return jsonify({"forecast": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Healthcheck
# ============================================================
@app.route("/healthcheck")
def healthcheck():
    return jsonify({"status": "OK"})


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
