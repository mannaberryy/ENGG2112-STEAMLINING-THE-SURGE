from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import pandas as pd

from backend.predict import generate_prediction

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

    # FIX: Save the uploaded CSV to the system temp directory instead of the
    # project folder. VS Code Live Server watches the project folder for changes
    # and reloads the browser page whenever any file inside it is created or
    # modified — including the uploaded CSV. Saving to the OS temp directory
    # (e.g. C:\Users\...\AppData\Local\Temp) is completely outside the project,
    # so Live Server never sees the write and the page is not reloaded.
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
        # Always clean up the temp file, even if prediction fails
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(records), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=False)