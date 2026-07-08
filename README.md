# XAI Alert Triage

![CI](https://github.com/shri-j41/xai-alert-triage/actions/workflows/ci.yml/badge.svg)

An explainable, context-aware **security alert triage decision-support tool** for resource-constrained small and medium-sized enterprises (SMEs). It sits on top of existing SIEM tools (native Wazuh support, generic CSV import for others) and helps analysts answer three questions: *Which alert should be handled first? Why? What should I do next?*

> **Note:** This is a decision-support layer, not a replacement for a SIEM, EDR, or SOC platform. It does not perform automated response — the final triage decision always remains with the human analyst.

## How it works

Each alert passes through six layers:

1. **Alert Input** — simulated alerts, uploaded SIEM export CSV (auto-mapped columns), or Wazuh JSON logs (NDJSON, JSON array, or OpenSearch format)
2. **Context Enrichment** — adds SME-specific context: asset criticality, business function, internet exposure, user role, working hours
3. **Triage Scoring** — an XGBoost classifier assigns each alert a Low / Medium / High priority using six contextual features
4. **Explainability** — SHAP (TreeExplainer) generates a plain-English explanation, e.g. *"ranked high because: alert severity (42%), critical asset (28%), attack category risk (26%)"*
5. **Triage Guidance** — rule-based recommended first actions per alert
6. **Human Review** — the analyst Accepts, Escalates, or Dismisses each alert, with optional notes, undo, and a full decision log

## Screens

- **Load** — pick an alert source and run triage
- **Queue** — priority-sorted alert cards (list view) or one-at-a-time focus mode, with SHAP score breakdowns, recommended actions, decision buttons, and filter pills (All / High / Medium / Low / Reviewed)
- **Impact** — evaluation dashboard: Kendall's Tau divergence from severity-only ranking, Precision@20 comparison, critical-asset elevation, explanation coverage, and the analyst decision log

## Evaluation results

Compared against a severity-only baseline on the same 200 synthetic alerts:

| Metric | Severity-only | Context-aware |
|---|---|---|
| Kendall's Tau (vs baseline) | — | τ = 0.27, p < 0.001 |
| Precision@20 | 40% | 70% |
| Critical-asset avg. rank | 93.4 | 83.9 (+9.5 positions earlier) |
| Explanation coverage | — | 100% |

## Getting started

```bash
git clone https://github.com/shri-j41/xai-alert-triage.git
cd xai-alert-triage
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, choose **Simulated (200 alerts)**, and click **Run Triage**.

## Run with Docker

```bash
docker build -t xai-alert-triage .
docker run -p 8501:8501 xai-alert-triage
```

Then open http://localhost:8501.

## Project structure

```
├── app.py                        # Streamlit UI (Load / Queue / Impact screens)
├── requirements.txt
├── Dockerfile                    # Container build (python 3.11-slim, Streamlit on 8501)
├── .dockerignore
├── .github/workflows/ci.yml      # CI — runs the pytest suite on every push
├── .streamlit/config.toml        # Forces light theme
├── test_pipeline.py              # Pytest suite: end-to-end pipeline (3 tests)
├── test_wazuh_parser.py          # Pytest suite: Wazuh parser + integration (4 tests)
├── data/
│   ├── generate_alerts.py        # Seeded (SEED=42) generator of 200 synthetic alerts
│   └── alerts_raw.csv            # The generated simulated dataset (Nepali SME context)
├── models/
│   └── xgb_triage.ubj            # Trained XGBoost model (native format)
└── modules/
    ├── wazuh_parser.py           # Wazuh JSON → standard alert table
    ├── csv_mapper.py             # Arbitrary CSV columns → expected schema
    ├── context_enricher.py       # SME context enrichment + coverage check
    ├── model.py                  # XGBoost training / loading / scoring
    ├── explainer.py              # SHAP explanations per alert
    ├── description_generator.py  # Fallback human-readable alert descriptions
    ├── guidance_generator.py     # Rule-based triage guidance per alert category
    └── evaluator.py              # Baseline-vs-proposed evaluation (τ, Precision@20, charts)
```

## Tech stack

Python · Streamlit · XGBoost · SHAP · pandas / NumPy · scikit-learn · scipy · matplotlib · pytest

## Data

The primary dataset is 200 synthetic Wazuh-style alerts modelling a Nepali SME environment (finance servers, ERP, POS terminals, domain controllers, web servers). Real organisational logs were not used due to ethical and access constraints. Generation is fully seeded, so results are reproducible.

## Running the tests

The test files are proper pytest suites (7 tests total) and are also runnable as plain scripts:

```bash
pytest test_pipeline.py test_wazuh_parser.py -v
# or
python test_pipeline.py
python test_wazuh_parser.py
```

Tests run automatically on every push via GitHub Actions.

## Limitations

- Evaluated on synthetic data; findings require validation on real logs before production use
- Not real-time — batch triage of loaded/exported alerts
- Decisions are held in session state; use **Export Results** to persist the decision log as CSV

## Author

**Shrijal Esmali** ([@shri-j41](https://github.com/shri-j41))
