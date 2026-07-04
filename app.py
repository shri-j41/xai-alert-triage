"""
Context-aware alert triage dashboard (Streamlit entry point).

Three screens, matching the Claude Design handoff:
  1. Load    — pick a source (Simulated / SIEM export CSV / SIEM logs JSON), run triage
  2. Queue   — priority-ordered review queue, List view or Focus mode
  3. Impact  — live evaluation of context-aware vs. severity-only ranking

The ML pipeline (enrich -> score -> explain) and the 200-alert synthetic
dataset are unchanged from the original prototype — only the UI/UX and the
session-state decision tracking are new. See modules/ for the preserved
logic; this file is orchestration + presentation only.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from data.generate_alerts import generate
from modules.context_enricher import enrich, enrichment_coverage
from modules.description_generator import get_display_description
from modules.model import MODEL_PATH, load_model, predict, train
from modules.explainer import build_explainer, explain_dataframe
from modules.csv_mapper import (
    REQUIRED_COLS,
    auto_map,
    apply_mapping,
    needs_manual_mapping,
)
from modules.wazuh_parser import parse_string
from modules.evaluator import evaluate, build_export_zip
from modules.guidance_generator import guidance_for

RAW_PATH = "data/alerts_raw.csv"

# Context-tag colour tokens, taken verbatim from the Claude Design handoff.
CONTEXT_STYLES = {
    "Critical asset":   ("#B42318", "#FEF3F2", "#ECAFA6"),
    "Internet-facing":  ("#B54708", "#FFFAEB", "#F5D9A8"),
    "Standard asset":   ("#3D4854", "#F6F7F9", "#CFD6DD"),
    "Internal-only":    ("#66707C", "#F6F7F9", "#E4E8ED"),
}
FOCUS_BADGE_STYLES = {
    "High":   ("#B42318", "#FFFFFF"),
    "Medium": ("#FFFAEB", "#B54708"),
    "Low":    ("#F0F2F5", "#66707C"),
}
TIER_KEY = {"High": "high", "Medium": "med", "Low": "low"}

st.set_page_config(page_title="Context-aware alert triage", page_icon="🛡️", layout="wide")


# ══════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "screen": "load",
        "alerts": None,          # scored + explained DataFrame (output of run_pipeline)
        "decisions": {},         # alert_id -> {action, timestamp, priority_at_decision, note}
        "current_index": 0,      # Focus-mode position within the current filtered queue
        "filter": "All",         # "All" / "High" / "Medium" / "Low"
        "view_mode": "list",     # "list" / "focus"
        "coverage_warning": None,  # enrichment-coverage warning text for uploaded data
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _set_screen(s: str) -> None:
    st.session_state.screen = s


def _set_view(v: str) -> None:
    st.session_state.view_mode = v
    st.session_state.current_index = 0


def _set_filter(f: str) -> None:
    st.session_state.filter = f
    st.session_state.current_index = 0


def _record_decision(alert_id: str, action: str, priority: str, note_key: str | None = None) -> None:
    note = str(st.session_state.get(note_key, "") or "").strip() if note_key else ""
    st.session_state.decisions[alert_id] = {
        "action": action,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "priority_at_decision": priority,
        "note": note,
    }


def _undo_decision(alert_id: str) -> None:
    st.session_state.decisions.pop(alert_id, None)


def _update_note(alert_id: str, note_key: str) -> None:
    """Add or edit the note on an already-recorded decision."""
    if alert_id in st.session_state.decisions:
        st.session_state.decisions[alert_id]["note"] = (
            str(st.session_state.get(note_key, "") or "").strip()
        )


def _focus_decide(alert_id: str, action: str, priority: str, n: int, note_key: str | None = None) -> None:
    # The Focus-mode form uses clear_on_submit, so the shared note input
    # resets itself after each decision — no manual clearing needed.
    _record_decision(alert_id, action, priority, note_key)
    st.session_state.current_index = (st.session_state.current_index + 1) % max(n, 1)


def _next_alert(n: int) -> None:
    st.session_state.current_index = (st.session_state.current_index + 1) % max(n, 1)


def _safe_key(value: str) -> str:
    """Sanitise an alert_id into a token safe for widget keys / CSS classes."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def _decisions_to_df() -> pd.DataFrame:
    if not st.session_state.decisions:
        return pd.DataFrame(columns=["alert_id", "action", "timestamp", "priority_at_decision", "note"])
    return pd.DataFrame(
        [{"alert_id": aid, **d} for aid, d in st.session_state.decisions.items()]
    )


# ══════════════════════════════════════════════════════════════════════════
# ML pipeline (preserved) — enrich -> score -> explain
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Preparing baseline XGBoost model...")
def _bootstrap_model():
    if not os.path.exists(RAW_PATH):
        raw = generate(n=200, out_path=RAW_PATH)
    else:
        raw = pd.read_csv(RAW_PATH)
    enriched = enrich(raw)
    if os.path.exists(MODEL_PATH):
        model = load_model()
    else:
        model = train(enriched)
    explainer = build_explainer(model)
    return model, explainer


def run_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """enrich -> score -> explain, for any canonical-schema DataFrame."""
    model, explainer = _bootstrap_model()
    enriched = enrich(df_raw)
    scored = predict(enriched, model)
    return explain_dataframe(explainer, scored)


# ══════════════════════════════════════════════════════════════════════════
# Small display helpers
# ══════════════════════════════════════════════════════════════════════════

def _fmt_time(ts) -> str:
    try:
        return pd.to_datetime(ts).strftime("%H:%M · %d %b %Y")
    except Exception:
        return str(ts)


def _context_tag(row) -> tuple[str, str, str, str]:
    crit = row.get("asset_criticality", 2)
    exposed = bool(row.get("internet_exposed", False))
    if crit >= 5:
        label = "Critical asset"
    elif exposed:
        label = "Internet-facing"
    elif crit <= 2:
        label = "Internal-only"
    else:
        label = "Standard asset"
    c, bg, border = CONTEXT_STYLES[label]
    return label, c, bg, border


def _fig_precision20(p20_base: float, p20_prop: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 1.7))
    labels = ["Severity-only", "Context-aware"]
    vals = [p20_base, p20_prop]
    colours = ["#CFD6DD", "#067647"]
    y = [0, 1]
    ax.barh(y, vals, height=0.5, color=colours)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=10, fontweight="bold", color="#1C2530")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Global CSS — design tokens from the Claude Design handoff
# ══════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] { background:#F6F7F9 !important; }
/* :where() keeps the icon exclusion at zero extra specificity, so class-level
   font rules (e.g. IBM Plex Mono chips) still win over this base rule. */
html, body, [data-testid="stAppViewContainer"] *:where(:not([data-testid="stIconMaterial"])) { font-family:'IBM Plex Sans', sans-serif; }
[data-testid="stHeader"] { display:none; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding:0 !important; max-width:100% !important; }

/* ---- top nav bar ---- */
[class*="st-key-header_bar"] {
  background:#FFFFFF; border-bottom:1px solid #E4E8ED; padding:10px 32px;
}
[class*="st-key-nav_"] .stButton>button {
  border:none !important; font-size:13px !important; font-weight:600 !important;
  border-radius:6px !important; box-shadow:none !important; white-space:nowrap !important;
}
[class*="st-key-nav_"] .stButton>button p { white-space:nowrap !important; }

/* ---- load screen ---- */
[class*="st-key-load_shell"] { max-width:460px; margin:56px auto 0; text-align:center; }
[class*="st-key-load_card"] {
  background:#FFFFFF; border:1px solid #E4E8ED; border-radius:12px; padding:32px;
  text-align:left; margin-top:24px; box-shadow:0 1px 4px rgba(28,37,48,0.04);
}
[class*="st-key-run_triage"] .stButton>button {
  background:#1C2530 !important; color:#FFFFFF !important; border:none !important;
  font-weight:600 !important; height:48px !important; border-radius:8px !important;
}

/* ---- queue top bar ---- */
[class*="st-key-queue_topbar"] { background:#FFFFFF; border-bottom:1px solid #E4E8ED; padding:14px 32px; }
.qt-status { font-size:14px; color:#3D4854; margin-bottom:10px; }
.qt-high { color:#B42318; font-weight:600; }
.qt-muted { color:#66707C; }
[class*="st-key-view_"] .stButton>button, [class*="st-key-filt_"] .stButton>button {
  border-radius:8px !important; font-size:13px !important; font-weight:600 !important;
  white-space:nowrap !important; padding:6px 12px !important;
}
[class*="st-key-filt_"] .stButton>button { border-radius:16px !important; border:1px solid #CFD6DD !important; }
[class*="st-key-goto_impact_link"] .stButton>button {
  background:transparent !important; border:none !important; color:#66707C !important;
  text-decoration:underline; font-weight:500 !important;
}

/* ---- list view cards ---- */
[class*="st-key-queue_list_shell"] { max-width:860px; margin:0 auto; padding:24px 8px 48px; }
.tri-top { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:8px; }
.tri-badge { display:inline-block; font-weight:700; letter-spacing:0.08em; border-radius:6px; padding:5px 12px; font-size:12px; }
.tri-badge-high { background:#B42318; color:#FFFFFF; }
.tri-badge-med { background:#FFFAEB; color:#B54708; border:1px solid #F5D9A8; }
.tri-badge-low { background:#F0F2F5; color:#66707C; font-size:11px; padding:3px 8px; }
.tri-title { font-weight:700; flex:1; min-width:200px; color:#1C2530; }
.tri-time { font-family:'IBM Plex Mono', monospace; font-size:12.5px; color:#66707C; }
.tri-tags { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:8px; }
.tri-chip { font-family:'IBM Plex Mono', monospace; font-size:12.5px; background:#F0F2F5; border-radius:6px; padding:4px 10px; color:#3D4854; }
.tri-ctx { font-size:12px; font-weight:600; border-radius:12px; padding:3px 10px; border:1px solid; }
.tri-desc { margin:0 0 6px; line-height:1.5; color:#1C2530; }
.tri-desc-low { color:#8B95A1; font-size:13px; }
.tri-guidance { margin:0 0 10px; font-size:13.5px; color:#66707C; }
.decided-chip { font-size:13px; font-weight:600; color:#067647; background:#ECFDF3; border-radius:12px; padding:4px 12px; display:inline-block; }
[class*="st-key-undo_"] button {
  background:rgba(28,37,48,0.06) !important; color:#3D4854 !important; border:none !important;
  font-size:12.5px !important; font-weight:600 !important; padding:5px 14px !important;
}
[class*="st-key-undo_"] button:hover { background:rgba(28,37,48,0.12) !important; }
[class*="st-key-save_"] button {
  background:#FFFFFF !important; color:#067647 !important; border:1px solid #ABEFC6 !important;
  font-size:12.5px !important; font-weight:600 !important; padding:5px 14px !important;
}
[class*="st-key-save_"] button:hover { background:#ECFDF3 !important; }
[class*="st-key-note_"] input, [class*="st-key-focus_note"] input, [class*="st-key-editnote_"] input {
  font-size:13px !important; background:#FFFFFF !important; border:1px solid #E4E8ED !important;
}
[class*="st-key-nform_"] { border:none !important; padding:0 !important; }

[class*="st-key-card_high_"] {
  background:#FFFFFF; border:1.5px solid #ECAFA6; border-left:5px solid #B42318;
  border-radius:12px; padding:22px 28px 14px; margin-bottom:16px;
  box-shadow:0 3px 14px rgba(180,35,24,0.08);
}
[class*="st-key-card_med_"] {
  background:#FFFFFF; border:1px solid #E4E8ED; border-left:4px solid #B54708;
  border-radius:12px; padding:18px 24px 10px; margin-bottom:14px;
}
[class*="st-key-card_low_"] {
  background:#FFFFFF; border:1px solid #E4E8ED; border-radius:10px;
  padding:12px 20px 8px; margin-bottom:10px; opacity:0.62;
}
[class*="st-key-accept_"] button, [class*="st-key-focus_accept"] button {
  background:#067647 !important; color:#FFFFFF !important; border:none !important; font-weight:600 !important;
}
[class*="st-key-accept_"] button p, [class*="st-key-focus_accept"] button p { color:#FFFFFF !important; }
[class*="st-key-escalate_"] button, [class*="st-key-focus_escalate"] button {
  background:#FFFFFF !important; color:#1C2530 !important; border:1px solid #CFD6DD !important; font-weight:600 !important;
}
[class*="st-key-dismiss_"] button, [class*="st-key-focus_dismiss"] button {
  background:transparent !important; color:#66707C !important; border:none !important; font-weight:500 !important;
}
[class*="st-key-dform_"] { border:none !important; padding:0 !important; }

/* ---- focus mode ---- */
.focus-shell { max-width:680px; margin:0 auto; padding-top:24px; }
.focus-progress-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
.focus-pos { font-size:13px; color:#66707C; font-weight:500; }
.focus-bar-track { width:180px; height:4px; background:#E4E8ED; border-radius:2px; overflow:hidden; }
.focus-bar-fill { height:100%; background:#1C2530; border-radius:2px; }
[class*="st-key-focus_card"] {
  background:#FFFFFF; border:1px solid #E4E8ED; border-radius:14px; padding:36px 40px;
  box-shadow:0 4px 20px rgba(28,37,48,0.06); max-width:680px; margin:0 auto 16px;
}
.focus-badge { display:inline-block; font-size:13px; font-weight:700; letter-spacing:0.1em; padding:7px 16px; border-radius:7px; margin-bottom:14px; }
.focus-title { margin:0 0 10px !important; font-size:28px !important; font-weight:700 !important; line-height:1.25 !important; color:#1C2530 !important; }
.focus-tags { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:16px; }
.focus-time { font-family:'IBM Plex Mono', monospace; font-size:12px; color:#66707C; }
.focus-why { font-size:17px; line-height:1.6; color:#1C2530; margin:0 0 6px; }
.focus-rec { background:#ECFDF3; border:1px solid #ABEFC6; border-radius:10px; padding:16px 20px; display:flex; flex-direction:column; gap:4px; margin:14px 0 18px; }
.focus-rec-label { font-size:12px; font-weight:700; color:#067647; letter-spacing:0.05em; }
.focus-rec-text { font-size:15px; color:#1C2530; line-height:1.5; }
.shap-row { display:grid; grid-template-columns:150px 1fr 44px; align-items:center; gap:12px; margin-bottom:8px; }
.shap-name { font-size:13px; color:#3D4854; }
.shap-track { height:8px; background:#E4E8ED; border-radius:4px; overflow:hidden; }
.shap-fill { height:100%; background:#3D4854; border-radius:4px; }
.shap-pct { font-family:'IBM Plex Mono', monospace; font-size:12.5px; color:#66707C; text-align:right; }

/* ---- impact screen ---- */
[class*="st-key-impact_shell"] { max-width:780px; margin:0 auto; padding:32px 8px 48px; }
.impact-card {
  background:#FFFFFF; border:1px solid #E4E8ED; border-radius:12px; padding:22px 24px;
  display:flex; flex-direction:column; gap:5px; margin-bottom:16px;
}
.impact-val { font-size:28px; font-weight:700; font-family:'IBM Plex Mono', monospace; color:#1C2530; }
.impact-label { font-size:14px; font-weight:600; color:#3D4854; }
.impact-sub { font-size:13px; color:#66707C; }
.impact-chart-title { font-size:14px; font-weight:600; color:#3D4854; margin:8px 0 4px; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    with st.container(key="header_bar"):
        n1, n2, n3, _sp = st.columns([1, 1, 1, 6])
        n1.button("Load", key="nav_load", on_click=_set_screen, kwargs={"s": "load"}, use_container_width=True)
        n2.button(
            "Queue", key="nav_queue", on_click=_set_screen, kwargs={"s": "queue"},
            use_container_width=True, disabled=st.session_state.alerts is None,
        )
        n3.button(
            "Impact", key="nav_impact", on_click=_set_screen, kwargs={"s": "impact"},
            use_container_width=True, disabled=st.session_state.alerts is None,
        )

    active = st.session_state.screen
    rules = []
    for key, name in (("nav_load", "load"), ("nav_queue", "queue"), ("nav_impact", "impact")):
        bg = "#F0F2F5" if active == name else "transparent"
        c = "#1C2530" if active == name else "#66707C"
        rules.append(f'[class*="st-key-{key}"] .stButton>button{{background:{bg} !important;color:{c} !important;}}')
    st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Screen 1 — Load
# ══════════════════════════════════════════════════════════════════════════

def render_load_screen() -> None:
    with st.container(key="load_shell"):
        st.markdown(
            """
            <h1 style="margin:0;font-size:26px;font-weight:700;color:#1C2530;">Start a triage session</h1>
            <p style="margin:8px 0 0;font-size:15px;color:#66707C;line-height:1.5;">
              Load your alerts. We rank them by what matters to your business.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="load_card"):
            source = st.selectbox(
                "Alert source",
                ["Simulated (200 alerts)", "SIEM export (CSV — any tool)", "SIEM logs (JSON — Wazuh format)"],
                key="load_source",
            )

            ready = True
            raw_upload = None
            mapping = None
            uploaded_wazuh = None

            if source == "SIEM export (CSV — any tool)":
                template_cols = [
                    "alert_id", "timestamp", "source_tool", "alert_category",
                    "severity", "affected_asset", "user", "source_ip", "alert_description",
                ]
                template_csv = pd.DataFrame(columns=template_cols).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV template", data=template_csv,
                    file_name="alert_template.csv", mime="text/csv", key="dl_template",
                )
                uploaded_csv = st.file_uploader("Upload alert CSV", type=["csv"], key="load_csv_file")
                ready = uploaded_csv is not None
                if uploaded_csv is not None:
                    raw_upload = pd.read_csv(uploaded_csv)
                    mapping = auto_map(raw_upload.columns.tolist())
                    unresolved = needs_manual_mapping(mapping)
                    if unresolved:
                        st.warning(
                            f"Could not auto-detect: **{', '.join(unresolved)}**. "
                            "Please map manually below."
                        )
                    with st.expander("Column mapping", expanded=bool(unresolved)):
                        cols_with_none = ["(none)"] + raw_upload.columns.tolist()
                        new_mapping = {}
                        for canon in REQUIRED_COLS:
                            current = mapping.get(canon)
                            idx0 = cols_with_none.index(current) if current in cols_with_none else 0
                            chosen = st.selectbox(
                                f"`{canon}`", cols_with_none, index=idx0, key=f"map_{canon}",
                            )
                            new_mapping[canon] = None if chosen == "(none)" else chosen
                        mapping = new_mapping

            elif source == "SIEM logs (JSON — Wazuh format)":
                uploaded_wazuh = st.file_uploader("Upload Wazuh JSON export", type=["json"], key="load_wazuh_file")
                ready = uploaded_wazuh is not None

            run_clicked = st.button(
                "Run Triage", key="run_triage", disabled=not ready, use_container_width=True,
            )

            if run_clicked:
                try:
                    if source == "Simulated (200 alerts)":
                        canonical = pd.read_csv(RAW_PATH) if os.path.exists(RAW_PATH) else generate(n=200, out_path=RAW_PATH)
                    elif source == "SIEM export (CSV — any tool)":
                        canonical = apply_mapping(raw_upload, mapping)
                    else:
                        content = uploaded_wazuh.read().decode("utf-8", errors="replace")
                        canonical = parse_string(content)

                    explained = run_pipeline(canonical)

                    # For uploaded CSVs, asset/user context comes from a
                    # hardcoded SME knowledge base (exact match -> keyword
                    # fallback -> generic default). Warn the analyst when some
                    # rows fell back to defaults, so context tags on those
                    # rows aren't mistaken for real intelligence.
                    st.session_state.coverage_warning = None
                    if source == "SIEM export (CSV — any tool)":
                        cov = enrichment_coverage(explained)
                        total = cov["total_rows"]
                        asset_t, user_t = cov["asset"], cov["user"]
                        if total and (asset_t["default"] > 0 or user_t["default"] > 0):
                            def _pct(count: int) -> float:
                                return 100 * count / total
                            st.session_state.coverage_warning = (
                                "Some uploaded asset/user values don't match the tool's known SME "
                                "profiles, so their context fields (criticality, business function, "
                                "internet exposure, user role) fell back to generic defaults. "
                                f"Assets: {_pct(asset_t['exact_match']):.0f}% exact match, "
                                f"{_pct(asset_t['keyword_fallback']):.0f}% keyword fallback, "
                                f"{_pct(asset_t['default']):.0f}% generic default · "
                                f"Users: {_pct(user_t['exact_match']):.0f}% exact match, "
                                f"{_pct(user_t['keyword_fallback']):.0f}% keyword fallback, "
                                f"{_pct(user_t['default']):.0f}% generic default. "
                                "Treat context tags on defaulted rows with caution."
                            )

                    st.session_state.alerts = explained
                    st.session_state.decisions = {}
                    st.session_state.current_index = 0
                    st.session_state.filter = "All"
                    st.session_state.view_mode = "list"
                    st.session_state.screen = "queue"
                    st.rerun()
                except ValueError as e:
                    st.error(f"Mapping error: {e}")
                except Exception as e:
                    st.error(f"Could not run triage: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Screen 2 — Queue
# ══════════════════════════════════════════════════════════════════════════

def _render_card(row: pd.Series, tier: str, i: int) -> None:
    aid = row["alert_id"]
    decided = aid in st.session_state.decisions

    with st.container(key=f"card_{tier}_{i}"):
        badge_text = row["priority_label"].upper()
        ctx_label, ctx_c, ctx_bg, ctx_border = _context_tag(row)
        asset_chip = f"{row['affected_asset']} · {row.get('business_function', 'Unknown')}"
        time_str = _fmt_time(row["timestamp"])
        desc, _ = get_display_description(row)
        guidance = guidance_for(row)
        title_size = {"high": "21px", "med": "16px", "low": "14px"}[tier]
        desc_class = "tri-desc tri-desc-low" if tier == "low" else "tri-desc"

        st.markdown(
            f"""
            <div class="tri-top">
              <span class="tri-badge tri-badge-{tier}">{badge_text}</span>
              <span class="tri-title" style="font-size:{title_size}">{row['alert_category']}</span>
              <span class="tri-time">{time_str}</span>
            </div>
            <div class="tri-tags">
              <span class="tri-chip">{asset_chip}</span>
              <span class="tri-ctx" style="color:{ctx_c};background:{ctx_bg};border-color:{ctx_border}">{ctx_label}</span>
            </div>
            <p class="{desc_class}">{desc}</p>
            <p class="tri-guidance"><b style="color:#3D4854">Suggested:</b> {guidance}</p>
            """,
            unsafe_allow_html=True,
        )

        if decided:
            dec = st.session_state.decisions[aid]
            time_part = dec["timestamp"][11:16] if len(dec.get("timestamp", "")) >= 16 else ""
            st.markdown(
                f'<span class="decided-chip">✓ {dec["action"]}</span>'
                f'<span style="font-size:12px;color:#8B95A1;margin-left:10px;">reviewed at {time_part}</span>',
                unsafe_allow_html=True,
            )
            edit_key = f"editnote_{_safe_key(aid)}"
            with st.form(key=f"nform_{tier}_{i}", border=False, enter_to_submit=False):
                c_note, c_save, c_undo = st.columns([4, 1, 1])
                c_note.text_input(
                    "Decision note", key=edit_key, value=dec.get("note", ""),
                    label_visibility="collapsed", placeholder="Add a note to this decision",
                )
                c_save.form_submit_button(
                    "Save", key=f"save_{_safe_key(aid)}", on_click=_update_note,
                    kwargs={"alert_id": aid, "note_key": edit_key},
                )
                c_undo.form_submit_button(
                    "Undo", key=f"undo_{_safe_key(aid)}", on_click=_undo_decision,
                    kwargs={"alert_id": aid},
                )
        else:
            # A form makes the decision + note atomic: on submit, Streamlit
            # sends the note input's current text together with the button
            # press, so the callback always sees the note — no blur/commit
            # race between typing and clicking.
            note_key = f"note_{_safe_key(aid)}"
            with st.form(key=f"dform_{tier}_{i}", clear_on_submit=True, border=False, enter_to_submit=False):
                b1, b2, b3, b_note = st.columns([1, 1, 1, 3])
                b1.form_submit_button(
                    "Accept", key=f"accept_{tier}_{i}", on_click=_record_decision,
                    kwargs={"alert_id": aid, "action": "Accepted", "priority": row["priority_label"], "note_key": note_key},
                )
                b2.form_submit_button(
                    "Escalate", key=f"escalate_{tier}_{i}", on_click=_record_decision,
                    kwargs={"alert_id": aid, "action": "Escalated", "priority": row["priority_label"], "note_key": note_key},
                )
                b3.form_submit_button(
                    "Dismiss", key=f"dismiss_{tier}_{i}", on_click=_record_decision,
                    kwargs={"alert_id": aid, "action": "Dismissed", "priority": row["priority_label"], "note_key": note_key},
                )
                b_note.text_input(
                    "Analyst note", key=note_key, label_visibility="collapsed",
                    placeholder="Note (optional)",
                )


def _empty_filter_message() -> str:
    if st.session_state.filter == "Reviewed":
        return "You haven't reviewed any alerts yet — decisions you make will appear here."
    return "No alerts match this filter."


def render_list_view(queue_df: pd.DataFrame) -> None:
    if queue_df.empty:
        st.info(_empty_filter_message())
        return
    for i, (_, row) in enumerate(queue_df.iterrows()):
        _render_card(row, TIER_KEY[row["priority_label"]], i)

    st.button("View impact metrics →", key="goto_impact_link", on_click=_set_screen, kwargs={"s": "impact"})


def render_focus_view(queue_df: pd.DataFrame) -> None:
    if queue_df.empty:
        st.info(_empty_filter_message())
        return

    n = len(queue_df)
    idx = st.session_state.current_index % n
    row = queue_df.iloc[idx]
    aid = row["alert_id"]
    decided = aid in st.session_state.decisions

    st.markdown(
        f"""
        <div class="focus-shell">
          <div class="focus-progress-row">
            <span class="focus-pos">Alert {idx + 1} of {n}</span>
            <div class="focus-bar-track"><div class="focus-bar-fill" style="width:{(idx + 1) / n * 100:.1f}%"></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="focus_card"):
        badge_text = row["priority_label"].upper() + " PRIORITY"
        badge_bg, badge_c = FOCUS_BADGE_STYLES[row["priority_label"]]
        ctx_label, ctx_c, ctx_bg, ctx_border = _context_tag(row)
        asset_chip = f"{row['affected_asset']} · {row.get('business_function', 'Unknown')}"
        time_str = _fmt_time(row["timestamp"])
        why, _ = get_display_description(row)
        guidance = guidance_for(row)

        st.markdown(
            f"""
            <span class="focus-badge" style="background:{badge_bg};color:{badge_c}">{badge_text}</span>
            <h2 class="focus-title">{row['alert_category']}</h2>
            <div class="focus-tags">
              <span class="tri-chip">{asset_chip}</span>
              <span class="tri-ctx" style="color:{ctx_c};background:{ctx_bg};border-color:{ctx_border}">{ctx_label}</span>
              <span class="focus-time">{time_str}</span>
            </div>
            <p class="focus-why">{why}</p>
            """,
            unsafe_allow_html=True,
        )

        contribs = row.get("shap_contributions") or {}
        with st.expander("See score breakdown"):
            if contribs:
                rows_html = "".join(
                    f"""<div class="shap-row">
                          <span class="shap-name">{name}</span>
                          <div class="shap-track"><div class="shap-fill" style="width:{pct}%"></div></div>
                          <span class="shap-pct">{pct:.0f}%</span>
                        </div>"""
                    for name, pct in contribs.items()
                )
                st.markdown(rows_html, unsafe_allow_html=True)
            else:
                st.caption("No SHAP breakdown available for this alert.")

        st.markdown(
            f"""
            <div class="focus-rec">
              <span class="focus-rec-label">RECOMMENDED ACTION</span>
              <span class="focus-rec-text">{guidance}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if decided:
            dec = st.session_state.decisions[aid]
            time_part = dec["timestamp"][11:16] if len(dec.get("timestamp", "")) >= 16 else ""
            st.markdown(
                f'<span class="decided-chip">✓ {dec["action"]}</span>'
                f'<span style="font-size:12px;color:#8B95A1;margin-left:10px;">reviewed at {time_part}</span>',
                unsafe_allow_html=True,
            )
            edit_key = f"editnote_focus_{_safe_key(aid)}"
            with st.form(key="nform_focus", border=False, enter_to_submit=False):
                c_note, c_save, c_undo = st.columns([4, 1, 1])
                c_note.text_input(
                    "Decision note", key=edit_key, value=dec.get("note", ""),
                    label_visibility="collapsed", placeholder="Add a note to this decision",
                )
                c_save.form_submit_button(
                    "Save", key="save_focus", on_click=_update_note,
                    kwargs={"alert_id": aid, "note_key": edit_key},
                )
                c_undo.form_submit_button(
                    "Undo", key="undo_focus", on_click=_undo_decision,
                    kwargs={"alert_id": aid},
                )
        else:
            # Form = atomic note + decision (see list-view comment).
            with st.form(key="dform_focus", clear_on_submit=True, border=False, enter_to_submit=False):
                st.text_input(
                    "Analyst note", key="focus_note", label_visibility="collapsed",
                    placeholder="Note (optional) — recorded with your decision",
                )
                b1, b2, b3, _sp = st.columns([1, 1, 1, 2])
                b1.form_submit_button(
                    "Accept", key="focus_accept", on_click=_focus_decide,
                    kwargs={"alert_id": aid, "action": "Accepted", "priority": row["priority_label"], "n": n, "note_key": "focus_note"},
                )
                b2.form_submit_button(
                    "Escalate", key="focus_escalate", on_click=_focus_decide,
                    kwargs={"alert_id": aid, "action": "Escalated", "priority": row["priority_label"], "n": n, "note_key": "focus_note"},
                )
                b3.form_submit_button(
                    "Dismiss", key="focus_dismiss", on_click=_focus_decide,
                    kwargs={"alert_id": aid, "action": "Dismissed", "priority": row["priority_label"], "n": n, "note_key": "focus_note"},
                )

    _nc1, nc2 = st.columns([4, 1])
    nc2.button("Next alert →", key="focus_next", on_click=_next_alert, kwargs={"n": n}, use_container_width=True)


def render_queue_screen() -> None:
    if st.session_state.alerts is None:
        st.info("Load alerts first.")
        if st.button("Go to Load screen", key="queue_goload"):
            st.session_state.screen = "load"
            st.rerun()
        return

    alerts = st.session_state.alerts
    decisions = st.session_state.decisions
    reviewed = len(decisions)
    high_total = int((alerts["priority_label"] == "High").sum())
    high_decided = sum(1 for d in decisions.values() if d["priority_at_decision"] == "High")
    high_remaining = max(high_total - high_decided, 0)
    total_remaining = max(len(alerts) - reviewed, 0)

    with st.container(key="queue_topbar"):
        st.markdown(
            f'<div class="qt-status"><b>{reviewed} reviewed</b>'
            f' · <span class="qt-high">{high_remaining} high-priority remaining</span>'
            f' · <span class="qt-muted">{total_remaining} to go</span></div>',
            unsafe_allow_html=True,
        )
        col_view, col_filter = st.columns([2, 5])
        with col_view:
            v1, v2 = st.columns(2)
            v1.button("List view", key="view_list", on_click=_set_view, kwargs={"v": "list"}, use_container_width=True)
            v2.button("Focus mode", key="view_focus", on_click=_set_view, kwargs={"v": "focus"}, use_container_width=True)
        with col_filter:
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.button("All", key="filt_all", on_click=_set_filter, kwargs={"f": "All"}, use_container_width=True)
            f2.button("High", key="filt_high", on_click=_set_filter, kwargs={"f": "High"}, use_container_width=True)
            f3.button("Medium", key="filt_med", on_click=_set_filter, kwargs={"f": "Medium"}, use_container_width=True)
            f4.button("Low", key="filt_low", on_click=_set_filter, kwargs={"f": "Low"}, use_container_width=True)
            f5.button(f"Reviewed ({reviewed})", key="filt_rev", on_click=_set_filter, kwargs={"f": "Reviewed"}, use_container_width=True)

    active_view = st.session_state.view_mode
    active_filter = st.session_state.filter
    rules = []
    for key, name in (("view_list", "list"), ("view_focus", "focus")):
        bg = "#FFFFFF" if active_view == name else "transparent"
        c = "#1C2530" if active_view == name else "#66707C"
        rules.append(f'[class*="st-key-{key}"] .stButton>button{{background:{bg} !important;color:{c} !important;border:none !important;}}')
    for key, name in (("filt_all", "All"), ("filt_high", "High"), ("filt_med", "Medium"), ("filt_low", "Low"), ("filt_rev", "Reviewed")):
        bg = "#1C2530" if active_filter == name else "#FFFFFF"
        c = "#FFFFFF" if active_filter == name else "#66707C"
        rules.append(f'[class*="st-key-{key}"] .stButton>button{{background:{bg} !important;color:{c} !important;}}')
    st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)

    if st.session_state.get("coverage_warning"):
        st.warning(st.session_state.coverage_warning)

    tier_rank = {"High": 0, "Medium": 1, "Low": 2}
    queue_df = alerts.copy()
    queue_df["_tier"] = queue_df["priority_label"].map(tier_rank)
    queue_df = queue_df.sort_values(["_tier", "priority_score"], ascending=[True, False]).reset_index(drop=True)
    if active_filter == "Reviewed":
        queue_df = queue_df[queue_df["alert_id"].isin(decisions.keys())].reset_index(drop=True)
    elif active_filter != "All":
        queue_df = queue_df[queue_df["priority_label"] == active_filter].reset_index(drop=True)

    if active_view == "list":
        with st.container(key="queue_list_shell"):
            render_list_view(queue_df)
    else:
        render_focus_view(queue_df)


# ══════════════════════════════════════════════════════════════════════════
# Screen 3 — Impact
# ══════════════════════════════════════════════════════════════════════════

def render_impact_screen() -> None:
    if st.session_state.alerts is None:
        st.info("Load alerts first.")
        if st.button("Go to Load screen", key="impact_goload"):
            st.session_state.screen = "load"
            st.rerun()
        return

    with st.container(key="impact_shell"):
        st.markdown(
            """
            <h1 style="margin:0 0 4px;font-size:24px;font-weight:700;color:#1C2530;">Does context-aware ranking work?</h1>
            <p style="margin:0 0 24px;font-size:14.5px;color:#66707C;">Measured against severity-only ranking on the same alert set.</p>
            """,
            unsafe_allow_html=True,
        )

        result = evaluate(st.session_state.alerts)
        m = result.metrics

        tau = m["kendall_tau"]
        p_val = m["kendall_p_value"]
        p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
        sig_text = "statistically significant difference" if p_val < 0.05 else "not a statistically significant difference"

        n_total = len(st.session_state.alerts)
        n_explained = int(st.session_state.alerts["shap_explanation"].notna().sum()) if n_total else 0
        coverage_pct = (n_explained / n_total * 100) if n_total else 0

        cards = [
            (f"τ = {tau:.2f}", "Divergence from severity-only ranking", f"{p_text} — {sig_text}"),
            (f"{m['precision_at_20_proposed']:.0%}", "Precision@20 (context-aware)",
             f"{m['precision_at_20_proposed']:.0%} vs {m['precision_at_20_baseline']:.0%} for severity-only"),
            (f"{m['critical_asset_elevation']:+.1f}", "Critical assets surfaced earlier",
             "Positions gained in the queue, on average"),
            (f"{coverage_pct:.0f}%", "Explanation coverage",
             "Alerts with a plain-English SHAP explanation"),
        ]
        col_a, col_b = st.columns(2)
        col_c, col_d = st.columns(2)
        for col, (val, label, sub) in zip([col_a, col_b, col_c, col_d], cards):
            with col:
                st.markdown(
                    f"""
                    <div class="impact-card">
                      <span class="impact-val">{val}</span>
                      <span class="impact-label">{label}</span>
                      <span class="impact-sub">{sub}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="impact-chart-title">Ranking quality — Precision@20 comparison</div>', unsafe_allow_html=True)
        fig = _fig_precision20(m["precision_at_20_baseline"], m["precision_at_20_proposed"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        dec_df = _decisions_to_df()
        if not dec_df.empty:
            st.markdown(
                '<div class="impact-chart-title">Analyst decision log — alerts you reviewed this session</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(dec_df, use_container_width=True, hide_index=True)

        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.button("← Back to Queue", key="impact_back", on_click=_set_screen, kwargs={"s": "queue"}, use_container_width=True)
        bcol2.download_button(
            "Export Results",
            data=_decisions_to_df().to_csv(index=False).encode("utf-8"),
            file_name="triage_decisions.csv", mime="text/csv",
            key="impact_export", use_container_width=True,
        )
        with bcol3:
            with st.expander("Full Report"):
                st.dataframe(result.metrics_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download full evaluation report (ZIP)",
                    data=build_export_zip(result),
                    file_name="triage_evaluation_report.zip", mime="application/zip",
                    key="impact_full_zip",
                )


# ══════════════════════════════════════════════════════════════════════════
# App entry
# ══════════════════════════════════════════════════════════════════════════

render_header()

screen = st.session_state.screen
if screen == "load":
    render_load_screen()
elif screen == "queue":
    render_queue_screen()
elif screen == "impact":
    render_impact_screen()
