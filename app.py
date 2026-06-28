import io
import json
import os
import uuid
from datetime import datetime

import joblib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "maintenance-system-secret")

MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
INFO_PATH = "model_info.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(INFO_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    raise RuntimeError(
        "Trained model artifacts are missing. Please run train_model.py first to generate model.pkl, preprocessor.pkl and model_info.json."
    )

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
with open(INFO_PATH, "r", encoding="utf-8") as f:
    model_info = json.load(f)


def patch_legacy_sklearn_model(loaded_model):
    """Repair old tree estimators loaded by newer scikit-learn versions."""
    for estimator in getattr(loaded_model, "estimators_", []):
        if not hasattr(estimator, "monotonic_cst"):
            estimator.monotonic_cst = None


patch_legacy_sklearn_model(model)

FEATURE_FIELDS = [
    "Machine_Type",
    "Installation_Year",
    "Operational_Hours",
    "Temperature_C",
    "Vibration_mms",
    "Sound_dB",
    "Oil_Level_pct",
    "Coolant_Level_pct",
    "Power_Consumption_kW",
    "Last_Maintenance_Days_Ago",
    "Maintenance_History_Count",
    "Failure_History_Count",
    "AI_Supervision",
    "Error_Codes_Last_30_Days",
    "Remaining_Useful_Life_days",
    "Laser_Intensity",
    "Hydraulic_Pressure_bar",
    "Coolant_Flow_L_min",
    "Heat_Index",
    "AI_Override_Events",
]

CATEGORICAL_OPTIONS = model_info.get("machine_types", [
    "Mixer",
    "Industrial_Chiller",
    "Pick_and_Place",
    "Vision_System",
    "Shuttle_System",
    "Labeler",
    "Automated_Screwdriver",
    "Shrink_Wrapper",
    "Laser_Cutter",
    "CMM",
    "CNC_Lathe",
    "Dryer",
    "Valve_Controller",
    "Furnace",
    "Carton_Former",
    "Hydraulic_Press",
    "Compressor",
    "AGV",
    "Robot_Arm",
    "Conveyor_Belt",
    "Forklift_Electric",
    "Press_Brake",
    "Boiler",
    "Vacuum_Packer",
    "XRay_Inspector",
    "Crane",
    "3D_Printer",
    "Palletizer",
    "Grinder",
    "CNC_Mill",
    "Injection_Molder",
    "Heat_Exchanger",
    "Pump",
])

BOOLEAN_OPTIONS = ["True", "False"]


def parse_numeric(value):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return 0


def normalize_boolean(value):
    return 1 if str(value).strip().lower() in ["1", "true", "yes", "on"] else 0


def clamp_probability(value):
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, probability))


def get_failure_probability(input_transformed):
    raw_probabilities = model.predict_proba(input_transformed)[0]
    classes = list(getattr(model, "classes_", []))
    positive_index = classes.index(1) if 1 in classes else min(1, len(raw_probabilities) - 1)
    scores = [max(0.0, float(score)) for score in raw_probabilities]
    total_score = sum(scores)

    if total_score > 0:
        return clamp_probability(scores[positive_index] / total_score)

    return clamp_probability(scores[positive_index])


def build_input_vector(form_data):
    vector = []
    for field in FEATURE_FIELDS:
        raw_value = form_data.get(field, "")
        if field == "Machine_Type":
            vector.append(raw_value or CATEGORICAL_OPTIONS[0])
        elif field in ["AI_Supervision"]:
            vector.append(normalize_boolean(raw_value))
        else:
            vector.append(parse_numeric(raw_value))
    return vector


def get_risk_level(probability):
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.60:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    if probability >= 0.20:
        return "Low Risk"
    return "Healthy"


def get_maintenance_recommendation(risk_level):
    recommendations = {
        "Healthy": "Continue normal machine operation with routine monitoring.",
        "Low Risk": "Schedule a routine inspection within the next week.",
        "Medium Risk": "Perform preventive maintenance within a few days.",
        "High Risk": "Inspect bearings, lubrication, and cooling systems immediately.",
        "Critical": "Stop operation and perform urgent maintenance before restarting.",
    }
    return recommendations.get(risk_level, recommendations["Medium Risk"])


def humanize_field(label):
    return (
        label.replace("_", " ")
        .replace("pct", "%")
        .replace("mms", "mm/s")
        .replace(" C", " deg C")
        .replace(" L min", " L/min")
    )


def generate_pdf(report_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    title_style = ParagraphStyle(name="Title", fontSize=18, leading=22, spaceAfter=16)
    normal_style = ParagraphStyle(name="Normal", fontSize=10, leading=14)
    bold_style = ParagraphStyle(name="Bold", fontSize=11, leading=14, spaceAfter=8)

    story = []
    story.append(Paragraph("Preventive Maintenance Prediction Report", title_style))
    story.append(Paragraph(f"Report ID: {report_data['report_id']}", normal_style))
    story.append(Paragraph(f"Generated: {report_data['generated_at']}", normal_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Prediction Summary", bold_style))
    summary_table_data = [
        ["Machine Health Status:", report_data["health_status"]],
        ["Failure Prediction:", report_data["prediction_text"]],
        ["Failure Probability:", f"{report_data['failure_probability']:.2%}"],
        ["Confidence Score:", f"{report_data['confidence_score']:.2%}"],
        ["Risk Level:", report_data["risk_level"]],
        ["Maintenance Recommendation:", report_data["recommendation"]],
    ]
    summary_table = Table(summary_table_data, colWidths=[170, 330])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f4f4")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Machine Input Parameters", bold_style))
    table_data = [["Parameter", "Value"]]
    for label, value in report_data["fields"].items():
        table_data.append([humanize_field(label), str(value)])

    parameter_table = Table(table_data, colWidths=[220, 280])
    parameter_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f4f4")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(parameter_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Machine Learning Model Information", bold_style))
    story.append(Paragraph(f"Algorithm: {model_info['algorithm']}", normal_style))
    story.append(Paragraph(f"Model Accuracy: {model_info['metrics']['accuracy']:.2%}", normal_style))
    story.append(Paragraph(f"Dataset: {model_info['dataset_name']} ({model_info['num_records']} records)", normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    metrics = model_info["metrics"]
    confusion = metrics["confusion_matrix"]
    feature_importances = model_info.get("feature_importances", [])
    roc_data = model_info.get("roc_curve", {})
    return render_template(
        "about.html",
        dataset_name=model_info["dataset_name"],
        num_records=model_info["num_records"],
        num_features=model_info["num_features"],
        target_variable=model_info["target_variable"],
        algorithm=model_info["algorithm"],
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        roc_auc=metrics["roc_auc"],
        confusion_matrix=confusion,
        feature_importances=feature_importances,
        roc_data=roc_data,
    )


@app.route("/predict", methods=["GET"])
def predict():
    return render_template(
        "predict.html",
        features=FEATURE_FIELDS,
        machine_types=CATEGORICAL_OPTIONS,
        boolean_options=BOOLEAN_OPTIONS,
    )


@app.route("/result", methods=["POST"])
def result():
    try:
        user_input = {field: request.form.get(field, "") for field in FEATURE_FIELDS}
        vector = build_input_vector(request.form)
        import pandas as pd
        input_df = pd.DataFrame([vector], columns=FEATURE_FIELDS)
        input_transformed = preprocessor.transform(input_df)
        proba = get_failure_probability(input_transformed)
        prediction = 1 if proba >= 0.5 else 0
    except Exception as e:
        flash(f"Error during prediction: {str(e)}", "danger")
        return redirect(url_for("predict"))
    confidence = max(proba, 1 - proba)
    risk_level = get_risk_level(proba)
    recommendation = get_maintenance_recommendation(risk_level)
    health_status = "At Risk" if prediction == 1 else "Healthy"
    prediction_text = "Likely to fail within 7 days" if prediction == 1 else "Unlikely to fail within 7 days"

    return render_template(
        "result.html",
        user_input=user_input,
        prediction=prediction,
        prediction_text=prediction_text,
        failure_probability=proba,
        confidence_score=confidence,
        risk_level=risk_level,
        recommendation=recommendation,
        health_status=health_status,
    )


@app.route("/download-report", methods=["POST"])
def download_report():
    user_input = {field: request.form.get(field, "") for field in FEATURE_FIELDS}
    prediction = int(request.form.get("prediction", 0))
    failure_probability = clamp_probability(request.form.get("failure_probability", 0.0))
    confidence_score = clamp_probability(request.form.get("confidence_score", 0.0))
    risk_level = request.form.get("risk_level", "Healthy")
    recommendation = request.form.get("recommendation", "Continue normal machine operation.")
    health_status = request.form.get("health_status", "Healthy")
    prediction_text = request.form.get("prediction_text", "Unlikely to fail within 7 days")

    report_data = {
        "report_id": str(uuid.uuid4()).split("-")[0].upper(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fields": user_input,
        "health_status": health_status,
        "prediction_text": prediction_text,
        "failure_probability": failure_probability,
        "confidence_score": confidence_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
    }

    pdf_buffer = generate_pdf(report_data)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"maintenance_report_{report_data['report_id']}.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
