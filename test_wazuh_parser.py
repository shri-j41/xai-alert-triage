"""Tests all three Wazuh JSON input formats and the full pipeline integration.

Runs under pytest, or as a plain script (python test_wazuh_parser.py) for the
original smoke-test behaviour.
"""
import json
import os
import sys
sys.path.insert(0, ".")

from modules.wazuh_parser import parse_string
from modules.context_enricher import enrich
from modules.model import load_model, train, predict
from modules.explainer import build_explainer, explain_dataframe

# Canonical schema every parsed alert row must have (see wazuh_parser._parse_one)
EXPECTED_COLUMNS = {
    "alert_id", "timestamp", "source_tool", "alert_category", "severity",
    "affected_asset", "user", "source_ip", "alert_description",
}
VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}

# ── Format 1: NDJSON ─────────────────────────────────────────────────────────
NDJSON = "\n".join([
    '{"timestamp":"2024-03-15T14:23:45.123+0000","id":"1234567890.1","rule":{"level":10,"description":"Multiple failed SSH login attempts"},"agent":{"name":"web-server-01"},"data":{"srcip":"185.220.101.45","dstuser":"admin"}}',
    '{"timestamp":"2024-03-15T22:10:05.000+0000","id":"1234567890.2","rule":{"level":14,"description":"Ransomware file encryption detected"},"agent":{"name":"fin-server-01"},"data":{"srcip":"192.168.1.15","dstuser":"SYSTEM"}}',
    '{"timestamp":"2024-03-15T08:05:00.000+0000","id":"1234567890.3","rule":{"level":5,"description":"Firewall blocked connection"},"agent":{"name":"pos-terminal-01"},"data":{"srcip":"103.45.67.89"}}',
])

# ── Format 2: JSON array ──────────────────────────────────────────────────────
ARRAY_JSON = json.dumps([
    {"timestamp":"2024-04-01T09:00:00.000+0000","id":"AAA.1","rule":{"level":8,"description":"Brute force password attack detected"},"agent":{"name":"dc-server-01"},"data":{"srcip":"45.33.32.156"}},
    {"timestamp":"2024-04-01T23:55:00.000+0000","id":"AAA.2","rule":{"level":13,"description":"Privilege escalation via sudo"},"agent":{"name":"erp-server-01"},"data":{"dstuser":"fin_manager","srcip":"192.168.1.10"}},
])

# ── Format 3: OpenSearch wrapped ─────────────────────────────────────────────
OS_JSON = json.dumps({"hits":{"hits":[
    {"_source":{"timestamp":"2024-05-10T11:30:00.000+0000","id":"OS.1","rule":{"level":12,"description":"Lateral movement via SMB detected"},"agent":{"name":"workstation-acct-01"},"data":{"srcip":"192.168.1.50","dstuser":"sysadmin"}}},
]}})


def _assert_standard_schema(df):
    assert EXPECTED_COLUMNS <= set(df.columns), f"missing: {EXPECTED_COLUMNS - set(df.columns)}"
    assert set(df["severity"].unique()) <= VALID_SEVERITIES
    assert df["alert_id"].notna().all()
    assert (df["affected_asset"].astype(str).str.strip() != "").all()


def test_ndjson_parses_to_standard_schema():
    df = parse_string(NDJSON)
    assert len(df) == 3
    _assert_standard_schema(df)
    assert list(df["alert_id"]) == ["1234567890.1", "1234567890.2", "1234567890.3"]
    assert df.loc[1, "alert_category"] == "Ransomware Indicator"
    assert df.loc[1, "severity"] == "Critical"
    assert df.loc[0, "user"] == "admin"
    assert df.loc[2, "user"] == "unknown"  # no dstuser in the third alert


def test_json_array_parses_to_standard_schema():
    df = parse_string(ARRAY_JSON)
    assert len(df) == 2
    _assert_standard_schema(df)
    assert df.loc[0, "alert_category"] == "Brute Force Attack"
    assert df.loc[1, "alert_category"] == "Privilege Escalation"
    assert df.loc[1, "user"] == "fin_manager"


def test_opensearch_wrapped_parses_to_standard_schema():
    df = parse_string(OS_JSON)
    assert len(df) == 1
    _assert_standard_schema(df)
    assert df.loc[0, "alert_id"] == "OS.1"
    assert df.loc[0, "alert_category"] == "Lateral Movement"
    assert df.loc[0, "affected_asset"] == "workstation-acct-01"


def _get_model(enriched_fallback=None):
    # Use the saved model when present; on a fresh checkout (e.g. CI) train
    # one from the seeded simulated dataset instead of depending on test
    # ordering to have created it.
    if os.path.exists("models/xgb_triage.ubj"):
        return load_model()
    from data.generate_alerts import generate
    raw = generate(n=200, out_path="data/alerts_raw.csv")
    return train(enrich(raw))


def test_full_pipeline_on_parsed_wazuh_alerts():
    df = parse_string(NDJSON)
    enriched = enrich(df)
    model = _get_model()
    scored = predict(enriched, model)
    explainer = build_explainer(model)
    explained = explain_dataframe(explainer, scored)

    assert len(explained) == 3
    assert set(explained["priority_label"].unique()) <= {"Low", "Medium", "High"}
    assert explained["shap_explanation"].notna().all()
    assert (explained["shap_explanation"].astype(str).str.strip() != "").all()


if __name__ == "__main__":
    df1 = parse_string(NDJSON)
    print("=== Format 1: NDJSON ===")
    print(df1[["alert_id","timestamp","alert_category","severity","affected_asset","user"]].to_string())

    df2 = parse_string(ARRAY_JSON)
    print("\n=== Format 2: JSON array ===")
    print(df2[["alert_id","alert_category","severity","affected_asset","user"]].to_string())

    df3 = parse_string(OS_JSON)
    print("\n=== Format 3: OpenSearch wrapped ===")
    print(df3[["alert_id","alert_category","severity","affected_asset","user"]].to_string())

    print("\n=== Full pipeline on NDJSON data ===")
    enriched = enrich(df1)
    model = _get_model()
    explainer = build_explainer(model)
    scored = predict(enriched, model)
    explained = explain_dataframe(explainer, scored)
    for _, r in explained.iterrows():
        print(f"  {r['alert_id']} [{r['priority_label']}] {r['shap_explanation']}")
