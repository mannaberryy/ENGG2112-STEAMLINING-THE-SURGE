from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import pandas as pd

from backend.predict import generate_prediction
from backend.triage import run_hospital_decision_support

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "backend is working"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No CSV file uploaded"}), 400

    file = request.files["file"]

    # Save to system temp directory so Live Server never sees the file
    # and does not reload the page
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(tmp_fd)

    try:
        file.save(tmp_path)

        results = generate_prediction(tmp_path)

        if isinstance(results, pd.DataFrame):
            records = results.to_dict(orient="records")
        else:
            records = [results.to_dict()]

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(records), 200


@app.route("/surge", methods=["POST"])
def surge():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    results = run_hospital_decision_support(
        priority_4_patients=int(data.get("priority_4_patients", 0)),
        priority_3_patients=int(data.get("priority_3_patients", 0)),
        priority_2_patients=int(data.get("priority_2_patients", 0)),
        priority_1_patients=int(data.get("priority_1_patients", 0)),
        total_icu_beds=int(data.get("total_icu_beds", 0)),
        occupied_icu_beds=int(data.get("occupied_icu_beds", 0)),
        surge_condition=data.get("surge_condition", "Custom")
    )

    # Convert all numpy int64/float64 columns to native Python float so
    # Flask's jsonify does not throw a TypeError on serialisation.
    # This is the cause of the silent failure on the second button click.
    results_clean = results.copy()
    for col in results_clean.select_dtypes(include=["number"]).columns:
        results_clean[col] = results_clean[col].astype(float)

    return jsonify({
        "recommended_strategy": str(results_clean.iloc[0]["strategy"]),
        "results": results_clean.to_dict(orient="records")
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=False)