from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import pandas as pd

app = Flask(__name__)

# ── Artifact paths 
ADABOOST_PATH = "artifacts/models/adaboost_model.pkl"
XGBOOST_PATH  = "artifacts/models/xgboost_model.pkl"
SCALER_PATH   = "artifacts/processed/scaler.pkl"
FEATURES_PATH = "artifacts/processed/features.pkl"
METRICS_PATH  = "artifacts/models/metrics.json"

# ── Load all artifacts at startup
ada_model    = joblib.load(ADABOOST_PATH)
xgb_model    = joblib.load(XGBOOST_PATH)
scaler       = joblib.load(SCALER_PATH)
FEATURES     = joblib.load(FEATURES_PATH)

with open(METRICS_PATH) as f:
    MODEL_METRICS = json.load(f)

# ── Tenure binning 
TENURE_BINS   = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']

# ── Dropdown options for the form 
DROPDOWN_OPTIONS = {
    "gender":            ["Male", "Female"],
    "SeniorCitizen":     ["No", "Yes"],
    "Partner":           ["Yes", "No"],
    "Dependents":        ["Yes", "No"],
    "PhoneService":      ["Yes", "No"],
    "MultipleLines":     ["No", "Yes", "No phone service"],
    "InternetService":   ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":    ["No", "Yes", "No internet service"],
    "OnlineBackup":      ["No", "Yes", "No internet service"],
    "DeviceProtection":  ["No", "Yes", "No internet service"],
    "TechSupport":       ["No", "Yes", "No internet service"],
    "StreamingTV":       ["No", "Yes", "No internet service"],
    "StreamingMovies":   ["No", "Yes", "No internet service"],
    "Contract":          ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling":  ["Yes", "No"],
    "PaymentMethod":     [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ],
}


def bin_tenure(tenure_val: float) -> str:
    """Return the tenure_bin label matching the notebook's pd.cut logic."""
    series = pd.cut(
        pd.Series([tenure_val]),
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True
    )
    return str(series.iloc[0])


def build_feature_vector(form) -> np.ndarray:
    """Convert POST form data into the 34-column scaled feature vector."""
    vec = {f: 0.0 for f in FEATURES}

    # ── Numeric ─
    vec['SeniorCitizen']   = 1.0 if form.get('SeniorCitizen') == 'Yes' else 0.0
    vec['MonthlyCharges']  = float(form.get('MonthlyCharges', 0))
    vec['TotalCharges']    = float(form.get('TotalCharges', 0))

    # ── tenure → tenure_bin one-hot 
    tenure_val = float(form.get('tenure', 1))
    slab       = bin_tenure(tenure_val)          
    col_key    = f'tenure_bin_{slab}'           
    if col_key in vec:
        vec[col_key] = 1.0
    # if slab == '1-12' all tenure_bin cols stay 0 (dropped reference)

    # ── get_dummies drop_first=True one-hots 
    # gender → gender_Male (Female is dropped reference)
    vec['gender_Male'] = 1.0 if form.get('gender') == 'Male' else 0.0

    # Binary Yes/No cols — reference is 'No' (dropped)
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        key = f'{col}_Yes'
        if key in vec:
            vec[key] = 1.0 if form.get(col) == 'Yes' else 0.0

    # MultipleLines — reference is 'No' (dropped)
    ml_val = form.get('MultipleLines', 'No')
    if ml_val == 'No phone service':
        vec['MultipleLines_No phone service'] = 1.0
    elif ml_val == 'Yes':
        vec['MultipleLines_Yes'] = 1.0

    # InternetService — reference is 'DSL' (dropped)
    is_val = form.get('InternetService', 'DSL')
    if is_val == 'Fiber optic':
        vec['InternetService_Fiber optic'] = 1.0
    elif is_val == 'No':
        vec['InternetService_No'] = 1.0

    # Cols with 3 levels where 'No' is the reference (dropped)
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']:
        val = form.get(col, 'No')
        if val == 'No internet service':
            vec[f'{col}_No internet service'] = 1.0
        elif val == 'Yes':
            vec[f'{col}_Yes'] = 1.0

    # Contract — reference is 'Month-to-month' (dropped)
    ct_val = form.get('Contract', 'Month-to-month')
    if ct_val == 'One year':
        vec['Contract_One year'] = 1.0
    elif ct_val == 'Two year':
        vec['Contract_Two year'] = 1.0

    # PaymentMethod — reference is 'Bank transfer (automatic)' (dropped)
    pm_val = form.get('PaymentMethod', 'Bank transfer (automatic)')
    pm_map = {
        'Credit card (automatic)': 'PaymentMethod_Credit card (automatic)',
        'Electronic check':        'PaymentMethod_Electronic check',
        'Mailed check':            'PaymentMethod_Mailed check',
    }
    if pm_val in pm_map and pm_map[pm_val] in vec:
        vec[pm_map[pm_val]] = 1.0

    arr = np.array([vec[f] for f in FEATURES], dtype=np.float64).reshape(1, -1)
    return scaler.transform(arr)


@app.route("/", methods=["GET", "POST"])
def index():
    ada_result = xgb_result = None

    if request.method == "POST":
        try:
            scaled = build_feature_vector(request.form)

            ada_pred = int(ada_model.predict(scaled)[0])
            ada_prob = float(ada_model.predict_proba(scaled)[0][1])
            xgb_pred = int(xgb_model.predict(scaled)[0])
            xgb_prob = float(xgb_model.predict_proba(scaled)[0][1])

            ada_result = {
                "label":       "Churn" if ada_pred else "No Churn",
                "probability": round(ada_prob * 100, 1),
                "churn":       ada_pred == 1,
                "risk":        "High" if ada_prob > 0.7 else "Medium" if ada_prob > 0.4 else "Low",
            }
            xgb_result = {
                "label":       "Churn" if xgb_pred else "No Churn",
                "probability": round(xgb_prob * 100, 1),
                "churn":       xgb_pred == 1,
                "risk":        "High" if xgb_prob > 0.7 else "Medium" if xgb_prob > 0.4 else "Low",
            }

        except Exception as e:
            ada_result = xgb_result = {
                "error": str(e), "label": "Error",
                "probability": 0, "churn": False, "risk": "—"
            }

    return render_template(
        "index.html",
        options=DROPDOWN_OPTIONS,
        ada_result=ada_result,
        xgb_result=xgb_result,
        metrics=MODEL_METRICS,
    )


@app.route("/metrics")
def metrics():
    return jsonify(MODEL_METRICS)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": len(FEATURES), "models": ["adaboost", "xgboost"]})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
