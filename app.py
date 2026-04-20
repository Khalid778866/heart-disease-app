from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import tensorflow as tf
import shap
from lime import lime_tabular
import os

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",    "model4.h5")
SCALER_PATH = os.getenv("SCALER_PATH",   "scaler4.pkl")
BG_PATH     = os.getenv("BG_PATH",       "X_train_bg.npy")   # 100-sample background

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Age", "Sex", "Chest_pain_type", "BP", "Cholesterol",
    "FBS_over_120", "EKG_results", "Max_HR", "Exercise_angina",
    "ST_depression", "Slope_of_ST", "Number_of_vessels_fluro", "Thallium"
]

FEATURE_LABELS = {
    "Age":                    "Age",
    "Sex":                    "Sex",
    "Chest_pain_type":        "Chest Pain Type",
    "BP":                     "Blood Pressure",
    "Cholesterol":            "Cholesterol",
    "FBS_over_120":           "Fasting Blood Sugar",
    "EKG_results":            "EKG Results",
    "Max_HR":                 "Max Heart Rate",
    "Exercise_angina":        "Exercise Angina",
    "ST_depression":          "ST Depression",
    "Slope_of_ST":            "ST Slope",
    "Number_of_vessels_fluro":"Vessels (Fluoroscopy)",
    "Thallium":               "Thallium Scan",
}

# Categorical feature indices (used by LIME to treat them correctly)
CATEGORICAL_FEATURES = [1, 2, 5, 6, 8, 10, 11, 12]

# ── Load artefacts at startup ─────────────────────────────────────────────────
print("Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading scaler…")
scaler = joblib.load(SCALER_PATH)

print("Initialising SHAP KernelExplainer (TF 2.16+ compatible)…")
if os.path.exists(BG_PATH):
    bg_data = np.load(BG_PATH)          # shape (N, 13) — already scaled
    print(f"  Background data loaded: {bg_data.shape}")
else:
    # Fallback: zeros background if file not found
    print("  WARNING: X_train_bg.npy not found — using zeros background")
    bg_data = np.zeros((50, len(FEATURE_NAMES)))

# Replaced DeepExplainer with KernelExplainer to avoid TF graph errors
shap_explainer = shap.KernelExplainer(model.predict, bg_data)
print("SHAP explainer ready")

print("Initialising LIME LimeTabularExplainer...")
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=bg_data,
    feature_names=FEATURE_NAMES,
    class_names=["Absence", "Presence"],
    categorical_features=CATEGORICAL_FEATURES,
    mode="classification",
    random_state=42,
)
print("LIME explainer ready")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob >= 0.75:   return "High"
    if prob >= 0.50:   return "Moderate"
    if prob >= 0.30:   return "Low-Moderate"
    return "Low"


def _shap_explanation(scaled_input: np.ndarray) -> list:
    """SHAP KernelExplainer — returns per-feature attributions sorted by |value|."""
    # KernelExplainer handles data slightly differently
    shap_vals = shap_explainer.shap_values(scaled_input, silent=True)

    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[0]).flatten()
    else:
        sv = np.array(shap_vals).flatten()

    result = []
    for i, name in enumerate(FEATURE_NAMES):
        result.append({
            "feature":     FEATURE_LABELS[name],
            "feature_key": name,
            "shap_value":  round(float(sv[i]), 4),
        })

    result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return result


def _lime_prediction_fn(X: np.ndarray) -> np.ndarray:
    """Wrapper for LIME: returns [P(Absence), P(Presence)] for each row."""
    preds = model.predict(X, verbose=0).flatten()
    return np.column_stack([1.0 - preds, preds])


def _lime_explanation(scaled_input: np.ndarray) -> list:
    """LIME tabular — returns per-feature local linear weights sorted by |value|."""
    exp = lime_explainer.explain_instance(
        data_row=scaled_input[0],
        predict_fn=_lime_prediction_fn,
        num_features=len(FEATURE_NAMES),
        labels=(1,)  # <--- FIX: Explicitly request explanation for class 1 (Presence)
    )

    # local_exp[1] will now always exist, preventing the KeyError
    lime_weights = {feat_idx: weight for feat_idx, weight in exp.local_exp[1]}

    result = [
        {
            "feature":     FEATURE_LABELS[FEATURE_NAMES[i]],
            "feature_key": FEATURE_NAMES[i],
            "lime_value":  round(float(lime_weights.get(i, 0.0)), 4),
        }
        for i in range(len(FEATURE_NAMES))
    ]
    result.sort(key=lambda x: abs(x["lime_value"]), reverse=True)
    return result


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        raw    = np.array([[float(data[f]) for f in FEATURE_NAMES]])
        scaled = scaler.transform(raw)
        prob   = float(model.predict(scaled, verbose=0)[0][0])
        label  = "Presence" if prob >= 0.5 else "Absence"

        # XAI — SHAP values
        shap_data = _shap_explanation(scaled)
        for item in shap_data:
            item["raw_value"] = round(float(data[item["feature_key"]]), 2)
            
        # XAI — LIME values
        lime_data = _lime_explanation(scaled)
        for item in lime_data:
            item["raw_value"] = round(float(data[item["feature_key"]]), 2)

        ev = shap_explainer.expected_value
        base = float(np.mean(ev) if hasattr(ev, '__len__') else ev)

        return jsonify({
            "prediction":  label,
            "probability": round(prob * 100, 2),
            "risk_level":  _risk_level(prob),
            "shap_values": shap_data,
            "lime_values": lime_data,
            "base_value":  round(base * 100, 2),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)