from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import math
import os
import time
import json
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# ============================================================
# ✅ CLEAN FUNCTION
# ============================================================
def clean(value):
    try:
        if value is None:
            return 0
        if isinstance(value, float) and math.isnan(value):
            return 0
        return value
    except:
        return 0


# ============================================================
# ✅ LOAD ML MODELS
# ============================================================
with open("water_quality_model.pkl", "rb") as f:
    classifier_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

lstm_model = load_model("lstm_model.h5")
lstm_scaler = joblib.load("scaler.pkl")

# ============================================================
# ✅ LOAD DATASET
# ============================================================
df = pd.read_csv("water_quality_dataset.csv").replace({np.nan: 0})

# ============================================================
# ✅ INTERNAL DATASET INDEX
# ============================================================
DATASET_INDEX_FILE = "dataset_index.json"

def load_index():
    if not os.path.exists(DATASET_INDEX_FILE):
        return 0
    try:
        with open(DATASET_INDEX_FILE, "r") as f:
            return json.load(f).get("index", 0)
    except:
        return 0

def save_index(idx):
    with open(DATASET_INDEX_FILE, "w") as f:
        json.dump({"index": idx}, f)


# ============================================================
# ✅ API: GET ROW BY INDEX
# ============================================================
@app.route("/get_row/<int:index>", methods=["GET"])
def get_row(index):
    try:
        if index < 0 or index >= len(df):
            return jsonify({"error": "Index out of range"}), 404

        row = df.iloc[index]

        return jsonify({
            "index": index,
            "TDS": float(clean(row["TDS"])),
            "Turbidity": float(clean(row["Turbidity"]))
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ API: GET CURRENT DATASET ROW
# ============================================================
@app.route("/dataset/current", methods=["GET"])
def dataset_current():
    try:
        idx = load_index()
        if idx >= len(df):
            idx = len(df) - 1

        row = df.iloc[idx]
        return jsonify({
            "index": idx,
            "TDS": float(row["TDS"]),
            "Turbidity": float(row["Turbidity"])
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ API: GET NEXT DATASET ROW
# ============================================================
@app.route("/dataset/next", methods=["GET"])
def dataset_next():
    try:
        idx = load_index()
        next_idx = idx + 1

        # loop back if out of range
        if next_idx >= len(df):
            next_idx = 0

        save_index(next_idx)

        row = df.iloc[next_idx]

        return jsonify({
            "index": next_idx,
            "TDS": float(row["TDS"]),
            "Turbidity": float(row["Turbidity"])
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ API: RESET DATASET INDEX
# ============================================================
@app.route("/dataset/reset", methods=["POST"])
def dataset_reset():
    try:
        save_index(0)
        return jsonify({"message": "Dataset index reset to 0"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ AI PREDICT (CLASSIFICATION)
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        tds = clean(data.get("TDS"))
        turb = clean(data.get("Turbidity"))

        X = pd.DataFrame([[tds, turb]], columns=["TDS", "Turbidity"])

        pred_class = classifier_model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        probabilities = classifier_model.predict_proba(X)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        history_row = {
            "TDS": tds,
            "Turbidity": turb,
            "Prediction": pred_label,
            "Confidence": confidence,
            "Timestamp": int(time.time())
        }

        if not os.path.exists("history.csv"):
            pd.DataFrame(columns=history_row.keys()).to_csv("history.csv", index=False)

        df_hist = pd.read_csv("history.csv")
        df_hist = pd.concat([df_hist, pd.DataFrame([history_row])], ignore_index=True)
        df_hist.to_csv("history.csv", index=False)

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ LSTM FUTURE FORECAST
# ============================================================
@app.route("/predict_future", methods=["POST"])
def predict_future():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 5))

        window = 3
        recent = df[["TDS", "Turbidity"]].tail(window).values
        scaled = lstm_scaler.transform(recent)

        input_seq = np.array([scaled])
        predictions = []

        for _ in range(steps):
            scaled_pred = lstm_model.predict(input_seq, verbose=0)[0]
            real = lstm_scaler.inverse_transform([scaled_pred])[0]

            predictions.append({
                "TDS": float(real[0]),
                "Turbidity": float(real[1])
            })

            input_seq = np.array([np.vstack([input_seq[0][1:], scaled_pred])])

        return jsonify({"future_predictions": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ IOT UPDATE (ESP32)
# ============================================================
@app.route("/iot_update", methods=["POST"])
def iot_update():
    try:
        data = request.get_json()

        row = {
            "TDS": clean(data.get("tds")),
            "Turbidity": clean(data.get("turbidity")),
            "Safe": data.get("safe", False),
            "Timestamp": int(data.get("ts", time.time()))
        }

        with open("latest_iot.json", "w") as f:
            json.dump(row, f)

        if not os.path.exists("iot_history.json"):
            with open("iot_history.json", "w") as f:
                json.dump([], f)

        with open("iot_history.json", "r") as f:
            hist = json.load(f)

        hist.append(row)

        with open("iot_history.json", "w") as f:
            json.dump(hist, f)

        return jsonify({"msg": "IoT data received OK"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ IOT LATEST
# ============================================================
@app.route("/iot_latest", methods=["GET"])
def iot_latest():
    try:
        if not os.path.exists("latest_iot.json"):
            return jsonify({"error": "No IoT data yet"}), 404

        with open("latest_iot.json", "r") as f:
            return jsonify(json.load(f)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ IOT HISTORY
# ============================================================
@app.route("/iot_history", methods=["GET"])
def iot_history():
    try:
        if not os.path.exists("iot_history.json"):
            return jsonify([]), 200

        with open("iot_history.json", "r") as f:
            return jsonify(json.load(f)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ START SERVER
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
