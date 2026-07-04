"""
XGBoost-based alert priority scorer.

Trains on the enriched alerts DataFrame using a rule-derived label
(priority_score 1-3: Low / Medium / High) so the model learns patterns
consistent with established triage logic, then exposes predict() and
explain() for the Streamlit dashboard.
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb

# Use XGBoost's native binary format (avoids pickle serialisation warnings
# when the model is reloaded across XGBoost versions).
MODEL_PATH = "models/xgb_triage.ubj"

# Severity numeric mapping
SEVERITY_NUM = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# Category base-risk weights (reflects CVSS-like intuition)
CATEGORY_RISK = {
    "Ransomware Indicator":         4,
    "Data Exfiltration Attempt":    4,
    "Privilege Escalation":         4,
    "Lateral Movement":             4,
    "Malware Detected":             3,
    "Brute Force Attack":           3,
    "Suspicious Process Execution": 3,
    "Unauthorised File Access":     2,
    "Firewall Rule Violation":      2,
    "Authentication Failure":       2,
    "Reconnaissance / Port Scan":   2,
    "Policy Violation":             1,
}

ROLE_RISK = {"Privileged": 3, "Service": 2, "Standard": 1, "Guest": 2}

FEATURE_COLS = [
    "severity_num",
    "asset_criticality",
    "category_risk",
    "internet_exposed",
    "user_role_num",
    "outside_working_hours",
]


def _derive_label(row: pd.Series) -> int:
    """
    Deterministic triage label derived from domain rules.
    Returns 0=Low, 1=Medium, 2=High — used as training supervision.
    """
    score = (
        row["severity_num"] * 0.30
        + row["asset_criticality"] * 0.25
        + row["category_risk"] * 0.20
        + row["internet_exposed"] * 0.10
        + row["user_role_num"] * 0.10
        + row["outside_working_hours"] * 0.05
    )
    # Normalise to 0-1 range (max possible = 4*.3 + 5*.25 + 4*.2 + 1*.1 + 3*.1 + 1*.05 = 3.9)
    score /= 3.9
    if score >= 0.65:
        return 2  # High
    if score >= 0.38:
        return 1  # Medium
    return 0      # Low


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["severity_num"]     = feat["severity"].map(SEVERITY_NUM).fillna(2).astype(int)
    feat["category_risk"]    = feat["alert_category"].map(CATEGORY_RISK).fillna(2).astype(int)
    feat["user_role_num"]    = feat["user_role"].map(ROLE_RISK).fillna(1).astype(int)
    feat["internet_exposed"] = feat["internet_exposed"].astype(int)
    feat["outside_working_hours"] = feat["outside_working_hours"].astype(int)
    return feat


def train(df: pd.DataFrame) -> xgb.XGBClassifier:
    feat = _build_features(df)
    feat["label"] = feat.apply(_derive_label, axis=1)

    X = feat[FEATURE_COLS].values
    y = feat["label"].values

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"[model] Trained XGBoost on {len(df)} samples, saved -> {MODEL_PATH}")
    return model


def load_model() -> xgb.XGBClassifier:
    """Load the native XGBoost model."""
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        return model

    raise FileNotFoundError(
        f"No model found at {MODEL_PATH}. Call train() first."
    )


def predict(df: pd.DataFrame, model: xgb.XGBClassifier) -> pd.DataFrame:
    """
    Returns df with added columns:
      priority_label  : str  ("Low" / "Medium" / "High")
      priority_score  : float 0-1 (probability of the predicted class)
    """
    feat = _build_features(df)
    X = feat[FEATURE_COLS].values

    probs = model.predict_proba(X)
    labels_idx = np.argmax(probs, axis=1)
    label_map = {0: "Low", 1: "Medium", 2: "High"}

    out = df.copy()
    out["priority_label"] = [label_map[i] for i in labels_idx]
    out["priority_score"] = probs[np.arange(len(probs)), labels_idx].round(3)
    out["_feat"] = list(X)          # carry raw features for SHAP
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.generate_alerts import generate
    from modules.context_enricher import enrich

    raw = generate(n=200, out_path="data/alerts_raw.csv")
    enriched = enrich(raw)
    model = train(enriched)
    scored = predict(enriched, model)
    print(scored[["alert_id", "severity", "asset_criticality",
                   "priority_label", "priority_score"]].head(10).to_string())
