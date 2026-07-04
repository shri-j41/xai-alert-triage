"""End-to-end pipeline tests: generate -> enrich -> train -> score -> explain.

Runs under pytest, or as a plain script (python test_pipeline.py) for the
original smoke-test behaviour.
"""
import sys
sys.path.insert(0, ".")

from data.generate_alerts import generate
from modules.context_enricher import enrich
from modules.model import train, predict
from modules.explainer import build_explainer, explain_dataframe

VALID_LABELS = {"Low", "Medium", "High"}

# The pipeline is expensive (trains XGBoost, computes SHAP), so run it once
# and share the result across tests. Generation and training are seeded,
# so the output is deterministic.
_explained = None


def _get_explained():
    global _explained
    if _explained is None:
        raw = generate(n=200, out_path="data/alerts_raw.csv")
        enriched = enrich(raw)
        model = train(enriched)
        scored = predict(enriched, model)
        explainer = build_explainer(model)
        _explained = explain_dataframe(explainer, scored)
    return _explained


def test_pipeline_produces_200_scored_alerts():
    df = _get_explained()
    assert len(df) == 200
    assert "priority_score" in df.columns
    assert df["priority_score"].notna().all()
    assert df["alert_id"].is_unique


def test_priority_labels_are_low_medium_high_only():
    df = _get_explained()
    labels = set(df["priority_label"].dropna().unique())
    assert labels <= VALID_LABELS, f"unexpected labels: {labels - VALID_LABELS}"
    assert df["priority_label"].notna().all()


def test_every_alert_gets_shap_explanation():
    df = _get_explained()
    assert "shap_explanation" in df.columns
    assert df["shap_explanation"].notna().all()
    assert (df["shap_explanation"].astype(str).str.strip() != "").all()


if __name__ == "__main__":
    explained = _get_explained()
    print("Pipeline OK. Top 5 alerts by priority score:")
    top5 = explained.sort_values("priority_score", ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f"  {r['alert_id']} [{r['priority_label']}] {r['shap_explanation']}")
