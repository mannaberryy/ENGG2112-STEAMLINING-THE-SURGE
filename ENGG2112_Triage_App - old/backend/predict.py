# ============================================================
# IMPORTS
# ============================================================

import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# LOAD MODELS
# ============================================================

model1_pipeline = joblib.load("backend/models/model1_pipeline.pkl")
model1_features = joblib.load("backend/models/model1_full_features.pkl")
model1_threshold = joblib.load("backend/models/model1_threshold.pkl")

model2_pipeline = joblib.load("backend/models/model2_pipeline.pkl")

# Fix old sklearn LogisticRegression pickle issue for Model 2
final_model = model2_pipeline.steps[-1][1]
if final_model.__class__.__name__ == "LogisticRegression":
    final_model.multi_class = "auto"

model2_features = joblib.load("backend/models/model2_full_features.pkl")
model2_threshold = joblib.load("backend/models/model2_threshold.pkl")


# ============================================================
# HELPER FUNCTION
# ============================================================

def prepare_input_data(input_df, required_features):
    prepared_df = input_df.copy()

    for col in required_features:
        if col not in prepared_df.columns:
            prepared_df[col] = np.nan

    prepared_df = prepared_df[required_features]

    return prepared_df


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def generate_prediction(csv_path):
    input_df = pd.read_csv(csv_path)

    patient_ids = None

    if "patient_id" in input_df.columns:
        patient_ids = input_df["patient_id"]
        input_df = input_df.drop(columns=["patient_id"])

    # MODEL 1
    model1_input = prepare_input_data(input_df, model1_features)

    print("\n===== MODEL 1 DEBUG =====")
    print("Model 1 input shape:", model1_input.shape)
    print("Model 1 non-missing values:", model1_input.notna().sum(axis=1).values)
    print("Model 1 classes:", model1_pipeline.classes_)

    model1_probs = model1_pipeline.predict_proba(model1_input)
    severe_prob = model1_probs[:, 1]

    print("Model 1 probabilities:")
    print(model1_probs)

    model1_pred = np.where(
        severe_prob >= model1_threshold,
        "severe",
        "non_severe"
    )

    # MODEL 2
    model2_input = prepare_input_data(input_df, model2_features)

    print("\n===== MODEL 2 DEBUG =====")
    print("Model 2 input shape:", model2_input.shape)
    print("Model 2 non-missing values:", model2_input.notna().sum(axis=1).values)
    print("Model 2 classes:", model2_pipeline.classes_)

    model2_probs = model2_pipeline.predict_proba(model2_input)
    deterioration_prob = model2_probs[:, 1]

    print("Model 2 probabilities:")
    print(model2_probs)

    model2_pred = np.where(
        deterioration_prob >= model2_threshold,
        "high_risk",
        "low_risk"
    )

    # OUTPUT
    results_df = pd.DataFrame({
        "model1_severity_prediction": model1_pred,
        "model1_severe_probability": severe_prob,
        "model2_deterioration_prediction": model2_pred,
        "model2_deterioration_probability": deterioration_prob
    })

    if patient_ids is not None:
        results_df.insert(0, "patient_id", patient_ids)

    return results_df


# ============================================================
# TEST RUN
# ============================================================

if __name__ == "__main__":
    results = generate_prediction("test_patient.csv")

    print("\n===== FINAL RESULTS =====")
    print(results)