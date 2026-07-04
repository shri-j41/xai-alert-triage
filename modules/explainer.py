"""
SHAP-based explanation module.

For each alert, computes SHAP values for the predicted class and returns:
  - a human-readable plain-English sentence
  - a dict of {feature: percentage_contribution} for bar charts
"""

import numpy as np
import shap
import xgboost as xgb
import pandas as pd

FEATURE_COLS = [
    "severity_num",
    "asset_criticality",
    "category_risk",
    "internet_exposed",
    "user_role_num",
    "outside_working_hours",
]

FEATURE_LABELS = {
    "severity_num":           "alert severity",
    "asset_criticality":      "critical asset",
    "category_risk":          "attack category risk",
    "internet_exposed":       "internet-exposed asset",
    "user_role_num":          "privileged user",
    "outside_working_hours":  "outside working hours",
}

# Predicted class index -> the word used in the explanation sentence, so a
# Low-priority alert reads "ranked low because ..." rather than always "high".
_PRIORITY_WORD = {0: "low", 1: "medium", 2: "high"}


def build_explainer(model: xgb.XGBClassifier) -> shap.Explainer:
    """
    Create a TreeExplainer using the tree_path_dependent method (the
    default when no background data is supplied). This is exact for tree
    ensembles and needs no background sample — unlike the interventional
    method (passing `data=`), which is ~100x slower per row and makes
    large uploads (thousands+ of alerts) appear to hang.
    """
    return shap.TreeExplainer(model, feature_names=FEATURE_COLS)


def _summarise(vals: np.ndarray, priority_word: str) -> tuple[str, dict]:
    """
    Turn a single row's SHAP values (one per feature, already sliced to the
    predicted class) into a plain-English sentence + contribution dict.
    """
    # Keep only positive contributors (factors that pushed priority UP)
    positive_mask = vals > 0
    if not positive_mask.any():
        # Fall back to absolute values if all contributions are negative
        positive_mask = np.ones(len(vals), dtype=bool)

    pos_vals  = vals[positive_mask]
    pos_names = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if positive_mask[i]]

    total = pos_vals.sum()
    if total == 0:
        contributions = {n: round(1.0 / len(pos_names) * 100, 1) for n in pos_names}
    else:
        contributions = {
            FEATURE_LABELS.get(n, n): round(float(v / total) * 100, 1)
            for n, v in zip(pos_names, pos_vals)
        }

    # Sort by contribution descending, take top 3
    top3 = dict(sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:3])

    parts = [f"{label} ({pct:.0f}%)" for label, pct in top3.items()]
    sentence = f"This alert was ranked {priority_word} because: " + ", ".join(parts) + "."

    return sentence, contributions


def explain_single(
    explainer: shap.Explainer,
    x_row: np.ndarray,
    predicted_class: int,
) -> tuple[str, dict]:
    """
    Explain one alert.

    Parameters
    ----------
    explainer      : SHAP TreeExplainer
    x_row          : 1-D numpy array of feature values (length = len(FEATURE_COLS))
    predicted_class: 0=Low, 1=Medium, 2=High

    Returns
    -------
    (plain_english_sentence, {feature_label: pct_contribution})
    """
    shap_values = explainer(x_row.reshape(1, -1))
    # shap_values.values shape: (1, n_features, n_classes)
    vals = shap_values.values[0, :, predicted_class]
    return _summarise(vals, _PRIORITY_WORD.get(predicted_class, "medium"))


def explain_dataframe(
    explainer: shap.Explainer,
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add explanation columns to the full scored DataFrame.
    Adds: shap_explanation (str), shap_contributions (dict)

    Calls the SHAP explainer once on the whole feature matrix (vectorised)
    rather than once per row — with large uploads (thousands+ of rows),
    per-row explainer calls make this step take minutes; batching it is
    orders of magnitude faster.
    """
    rev_map = {"Low": 0, "Medium": 1, "High": 2}

    X = np.stack(scored_df["_feat"].to_numpy())
    shap_values = explainer(X)  # shape: (n_rows, n_features, n_classes)
    pred_classes = scored_df["priority_label"].map(rev_map).fillna(1).astype(int).to_numpy()

    explanations = []
    contributions_list = []

    for i, pred_class in enumerate(pred_classes):
        vals = shap_values.values[i, :, pred_class]
        sentence, contribs = _summarise(vals, _PRIORITY_WORD.get(int(pred_class), "medium"))
        explanations.append(sentence)
        contributions_list.append(contribs)

    out = scored_df.copy()
    out["shap_explanation"]   = explanations
    out["shap_contributions"] = contributions_list
    return out
