"""
src/predict.py
Inference module:
  - Single customer prediction with SHAP explanation
  - Batch CSV prediction
  - Business risk labeling
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)


def load_inference_artifacts(config: dict):
    """Load model, scaler, feature names, and SHAP explainer."""
    save_dir = config["models"]["save_dir"]

    model        = joblib.load(os.path.join(save_dir, "best_model.pkl"))
    scaler       = joblib.load(os.path.join(save_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(save_dir, "feature_names.pkl"))
    explainer    = joblib.load(os.path.join(save_dir, "shap_explainer.pkl"))

    logger.info("Inference artifacts loaded successfully")
    return model, scaler, feature_names, explainer


def preprocess_single(raw_input: dict, scaler, feature_names: list,
                       config: dict) -> pd.DataFrame:
    """
    Convert a raw input dict (as received from UI/API) into a
    model-ready feature vector.

    Args:
        raw_input: Dict with keys matching original feature names
        scaler: Fitted StandardScaler
        feature_names: List of features in training order

    Returns:
        pd.DataFrame with 1 row, ready for model.predict_proba()
    """
    row = {f: 0 for f in feature_names}

    # Direct mappings
    direct_fields = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "SeniorCitizen", "gender", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling"
    ]
    for field in direct_fields:
        if field in raw_input and field in row:
            row[field] = raw_input[field]

    # Derived feature
    tenure = raw_input.get("tenure", 1)
    total  = raw_input.get("TotalCharges", raw_input.get("MonthlyCharges", 0) * tenure)
    row["ChargesPerMonth"] = total / (tenure + 1)
    row["HighRiskFlag"] = int(
        raw_input.get("MonthlyCharges", 0) > 65 and tenure < 12
    )

    # One-hot: Contract
    contract_map = {
        "Month-to-month": "Contract_Month-to-month",
        "One year"       : "Contract_One year",
        "Two year"       : "Contract_Two year",
    }
    contract_val = raw_input.get("Contract", "Month-to-month")
    key = contract_map.get(contract_val)
    if key and key in row:
        row[key] = 1

    # One-hot: InternetService
    internet_map = {
        "Fiber optic": "InternetService_Fiber optic",
        "DSL"        : "InternetService_DSL",
        "No"         : "InternetService_No",
    }
    internet_val = raw_input.get("InternetService", "DSL")
    key = internet_map.get(internet_val)
    if key and key in row:
        row[key] = 1

    # One-hot: PaymentMethod
    for pm in ["Electronic check", "Mailed check",
               "Bank transfer (automatic)", "Credit card (automatic)"]:
        key = f"PaymentMethod_{pm}"
        if key in row:
            row[key] = int(raw_input.get("PaymentMethod", "") == pm)

    df_input = pd.DataFrame([row])[feature_names]

    # Scale numerical
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "ChargesPerMonth"]
                if c in df_input.columns]
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    return df_input


def predict_single(raw_input: dict, model, scaler, feature_names: list,
                   explainer, config: dict) -> dict:
    """
    Predict churn for a single customer and return full explanation.

    Returns:
        dict with probability, label, risk_level, shap_values, top_factors
    """
    threshold = config["evaluation"]["threshold"]
    X = preprocess_single(raw_input, scaler, feature_names, config)

    prob      = float(model.predict_proba(X)[0][1])
    predicted = int(prob >= threshold)

    # Risk label
    if prob >= 0.70:
        risk = "HIGH"
    elif prob >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # SHAP
    shap_vals = explainer.shap_values(X)[0]
    shap_series = pd.Series(shap_vals, index=feature_names)
    top_factors = shap_series.abs().nlargest(5)

    result = {
        "churn_probability" : round(prob * 100, 2),
        "predicted_churn"   : bool(predicted),
        "risk_level"        : risk,
        "shap_values"       : shap_vals,
        "top_factors"       : top_factors.index.tolist(),
        "top_shap_values"   : shap_vals[top_factors.index.map(
                                  lambda x: feature_names.index(x)
                              )].tolist(),
    }

    logger.info(
        f"Prediction: {risk} RISK | Churn prob: {prob*100:.1f}% | "
        f"Top factor: {top_factors.index[0]}"
    )
    return result


def predict_batch(csv_path: str, output_path: str,
                  model, scaler, feature_names: list,
                  explainer, config: dict) -> pd.DataFrame:
    """
    Batch predict for a CSV of raw customer records.
    Adds columns: churn_probability, predicted_churn, risk_level.

    Args:
        csv_path: Path to input CSV
        output_path: Where to save predictions CSV
    """
    logger.info(f"Batch prediction on: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    results = []
    for _, row in df_raw.iterrows():
        res = predict_single(row.to_dict(), model, scaler, feature_names, explainer, config)
        results.append({
            "churn_probability": res["churn_probability"],
            "predicted_churn"  : res["predicted_churn"],
            "risk_level"       : res["risk_level"],
        })

    df_out = pd.concat([df_raw.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    df_out.to_csv(output_path, index=False)
    logger.info(f"Batch predictions saved -> {output_path}")
    logger.info(
        f"Summary — High: {(df_out['risk_level']=='HIGH').sum()} | "
        f"Medium: {(df_out['risk_level']=='MEDIUM').sum()} | "
        f"Low: {(df_out['risk_level']=='LOW').sum()}"
    )
    return df_out


if __name__ == "__main__":
    config = load_config()
    model, scaler, features, explainer = load_inference_artifacts(config)

    # Example single prediction
    sample = {
        "tenure": 3, "MonthlyCharges": 80.0, "TotalCharges": 240.0,
        "SeniorCitizen": 0, "gender": 0, "Partner": 0, "Dependents": 0,
        "PhoneService": 1, "MultipleLines": 0, "OnlineSecurity": 0,
        "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0,
        "StreamingTV": 1, "StreamingMovies": 1, "PaperlessBilling": 1,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }

    result = predict_single(sample, model, scaler, features, explainer, config)
    print(f"\nChurn Probability : {result['churn_probability']}%")
    print(f"Predicted Churn   : {result['predicted_churn']}")
    print(f"Risk Level        : {result['risk_level']}")
    print(f"Top 5 Factors     : {result['top_factors']}")
