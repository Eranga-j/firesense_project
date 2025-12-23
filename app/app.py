from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model (path from project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "firesense_model.pkl")
model = joblib.load(MODEL_PATH)

# These must match what you used in training (feature_cols)
FEATURE_COLS = [
    "equipment_type", "install_year", "equipment_age_years", "manufacturer", "condition",
    "months_since_service", "service_interval_months", "fault_count", "service_quality_score",
    "humidity_level", "temperature_avg_c", "coastal_exposure", "daily_exposure_hours",
    "power_fluctuation_level", "technician_experience_years", "certification_status",
    "last_audit_score", "activation_count", "activated_before",
    "floor_level", "near_emergency_exit", "backup_power_available", "usage_level"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form values
    form = request.form

    # Build one-row dataframe for prediction
    row = {
        "equipment_type": form.get("equipment_type"),
        "install_year": int(form.get("install_year")),
        "equipment_age_years": int(form.get("equipment_age_years")),
        "manufacturer": form.get("manufacturer"),
        "condition": form.get("condition"),

        "months_since_service": int(form.get("months_since_service")),
        "service_interval_months": int(form.get("service_interval_months")),
        "fault_count": int(form.get("fault_count")),
        "service_quality_score": int(form.get("service_quality_score")),

        "humidity_level": form.get("humidity_level"),
        "temperature_avg_c": float(form.get("temperature_avg_c")),
        "coastal_exposure": form.get("coastal_exposure"),
        "daily_exposure_hours": float(form.get("daily_exposure_hours")),

        "power_fluctuation_level": form.get("power_fluctuation_level"),
        "technician_experience_years": int(form.get("technician_experience_years")),
        "certification_status": form.get("certification_status"),

        "last_audit_score": int(form.get("last_audit_score")),
        "activation_count": int(form.get("activation_count")),
        "activated_before": form.get("activated_before"),

        "floor_level": int(form.get("floor_level")),
        "near_emergency_exit": form.get("near_emergency_exit"),
        "backup_power_available": form.get("backup_power_available"),
        "usage_level": form.get("usage_level")
    }

    X = pd.DataFrame([row], columns=FEATURE_COLS)

    pred = model.predict(X)[0]  # "Low" / "Medium" / "High"

    # simple color mapping for UI
    color = {"Low": "green", "Medium": "orange", "High": "red"}.get(pred, "black")

    return render_template("result.html", prediction=pred, color=color, inputs=row)

if __name__ == "__main__":
    app.run(debug=True)
