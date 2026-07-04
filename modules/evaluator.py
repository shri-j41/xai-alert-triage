"""
Evaluation module — Baseline vs. Proposed (context-aware XGBoost).

Approach 1 — Baseline (severity-only)
    Alerts ranked purely by numeric severity score:
        Critical=4, High=3, Medium=2, Low=1
    Ties broken by alert_id (stable, reproducible).
    No SME context is considered.

Approach 2 — Proposed (context-aware XGBoost)
    Alerts ranked by priority_score produced by the XGBoost pipeline,
    which incorporates asset criticality, user role, internet exposure,
    attack category risk, and temporal context.

Metrics computed
----------------
    kendall_tau             : rank-order correlation between the two rankings
                              (tau≈0 → very different; tau≈1 → identical)
    kendall_p               : two-sided p-value for H0: tau=0
    precision_at_10/20      : of top-K alerts, fraction with asset_criticality >= 4
    avg_rank_critical        : mean rank position of alerts on critical assets
                              (asset_criticality == 5); lower = ranked earlier
    avg_rank_exposed         : same for internet-exposed alerts
    avg_rank_outside_hours   : same for outside-working-hours alerts
    critical_elevation       : baseline avg_rank_critical − proposed avg_rank_critical
                              (positive = proposed surfaces critical alerts earlier)

Figures produced
----------------
    fig_precision   : grouped bar chart — Precision@10 and Precision@20
    fig_elevation   : grouped bar chart — avg rank position for three subgroups
    fig_scatter     : scatter plot — baseline rank vs proposed rank per alert
                      (points above the diagonal moved UP in the proposed ranking)

Returns
-------
    EvalResult dataclass with all of the above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────

SEVERITY_NUM = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

BASELINE_COLOUR  = "#5b7fbd"   # muted blue
PROPOSED_COLOUR  = "#d62728"   # thesis red
NEUTRAL_COLOUR   = "#aaaaaa"

# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    metrics:         dict[str, Any]
    baseline_ranked: pd.DataFrame        # all alerts, baseline order, rank column
    proposed_ranked: pd.DataFrame        # all alerts, proposed order, rank column
    comparison_top20: pd.DataFrame       # side-by-side top 20
    metrics_df:      pd.DataFrame        # flat metrics for CSV export
    fig_precision:   plt.Figure
    fig_elevation:   plt.Figure
    fig_scatter:     plt.Figure


# ── Core evaluation function ─────────────────────────────────────────────────

def evaluate(df_explained: pd.DataFrame) -> EvalResult:
    """
    Run the full evaluation on an explained alerts DataFrame.

    Parameters
    ----------
    df_explained : pd.DataFrame
        Output of run_pipeline() — must contain at minimum:
        alert_id, severity, asset_criticality, internet_exposed,
        outside_working_hours, priority_score.

    Returns
    -------
    EvalResult
    """
    df = df_explained.copy().reset_index(drop=True)

    # ── 1. Build baseline ranking ─────────────────────────────────────────────
    df["baseline_score"] = df["severity"].map(SEVERITY_NUM).fillna(2).astype(float)

    # Stable tiebreak: alerts at the same severity keep their original order,
    # which mirrors the FIFO queue behaviour of a non-triaged inbox.
    df_baseline = (
        df.sort_values("baseline_score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    df_baseline["baseline_rank"] = np.arange(1, len(df_baseline) + 1)

    # ── 2. Build proposed ranking ─────────────────────────────────────────────
    df_proposed = (
        df.sort_values("priority_score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    df_proposed["proposed_rank"] = np.arange(1, len(df_proposed) + 1)

    # ── 3. Merge ranks on alert_id ────────────────────────────────────────────
    rank_df = df_baseline[["alert_id", "baseline_rank", "baseline_score"]].merge(
        df_proposed[["alert_id", "proposed_rank", "priority_score",
                      "asset_criticality", "internet_exposed",
                      "outside_working_hours", "severity",
                      "alert_category", "affected_asset"]],
        on="alert_id",
    )

    # ── 4. Kendall's Tau ──────────────────────────────────────────────────────
    tau, p_val = stats.kendalltau(rank_df["baseline_rank"], rank_df["proposed_rank"])

    # ── 5. Precision@K ───────────────────────────────────────────────────────
    def _precision_at_k(ranked_df: pd.DataFrame, k: int, rank_col: str) -> float:
        top_k = ranked_df.nsmallest(k, rank_col)
        return float((top_k["asset_criticality"] >= 4).sum()) / k

    p10_base = _precision_at_k(rank_df, 10, "baseline_rank")
    p20_base = _precision_at_k(rank_df, 20, "baseline_rank")
    p10_prop = _precision_at_k(rank_df, 10, "proposed_rank")
    p20_prop = _precision_at_k(rank_df, 20, "proposed_rank")

    # ── 6. Average rank elevation for context subgroups ──────────────────────
    def _avg_rank(rank_df: pd.DataFrame, mask: pd.Series, rank_col: str) -> float:
        sub = rank_df[mask]
        return float(sub[rank_col].mean()) if len(sub) > 0 else float("nan")

    crit_mask    = rank_df["asset_criticality"] == 5
    exposed_mask = rank_df["internet_exposed"].astype(bool)
    outside_mask = rank_df["outside_working_hours"].astype(bool)

    avg_rank_crit_base    = _avg_rank(rank_df, crit_mask,    "baseline_rank")
    avg_rank_crit_prop    = _avg_rank(rank_df, crit_mask,    "proposed_rank")
    avg_rank_exp_base     = _avg_rank(rank_df, exposed_mask, "baseline_rank")
    avg_rank_exp_prop     = _avg_rank(rank_df, exposed_mask, "proposed_rank")
    avg_rank_out_base     = _avg_rank(rank_df, outside_mask, "baseline_rank")
    avg_rank_out_prop     = _avg_rank(rank_df, outside_mask, "proposed_rank")

    critical_elevation    = avg_rank_crit_base - avg_rank_crit_prop
    exposed_elevation     = avg_rank_exp_base  - avg_rank_exp_prop
    outside_elevation     = avg_rank_out_base  - avg_rank_out_prop

    metrics = {
        "kendall_tau":                    round(tau,   4),
        "kendall_p_value":                round(p_val, 4),
        "precision_at_10_baseline":       round(p10_base, 4),
        "precision_at_10_proposed":       round(p10_prop, 4),
        "precision_at_20_baseline":       round(p20_base, 4),
        "precision_at_20_proposed":       round(p20_prop, 4),
        "avg_rank_critical_baseline":     round(avg_rank_crit_base, 2),
        "avg_rank_critical_proposed":     round(avg_rank_crit_prop, 2),
        "avg_rank_exposed_baseline":      round(avg_rank_exp_base,  2),
        "avg_rank_exposed_proposed":      round(avg_rank_exp_prop,  2),
        "avg_rank_outside_hrs_baseline":  round(avg_rank_out_base,  2),
        "avg_rank_outside_hrs_proposed":  round(avg_rank_out_prop,  2),
        "critical_asset_elevation":       round(critical_elevation, 2),
        "exposed_asset_elevation":        round(exposed_elevation,  2),
        "outside_hours_elevation":        round(outside_elevation,  2),
        "n_alerts":                       len(df),
        "n_critical_assets":              int(crit_mask.sum()),
        "n_internet_exposed":             int(exposed_mask.sum()),
        "n_outside_hours":                int(outside_mask.sum()),
    }

    # ── 7. Comparison table — top 20 side by side ────────────────────────────
    # Use distinct column names "baseline_rank_pos" / "proposed_rank_pos" so
    # that after pd.concat(axis=1) there are NO duplicate column names.
    # Duplicate column names cause pandas Styler to raise
    # "non-unique index or columns" even when the row index is clean.
    top20_base = (
        df_baseline.head(20)
        [["baseline_rank", "alert_id", "severity", "alert_category",
          "affected_asset", "asset_criticality", "internet_exposed",
          "outside_working_hours"]]
        .rename(columns=lambda c: f"baseline_{c}" if c != "baseline_rank" else "baseline_rank_pos")
        .reset_index(drop=True)
    )
    top20_prop = (
        df_proposed.head(20)
        [["proposed_rank", "alert_id", "severity", "alert_category",
          "affected_asset", "asset_criticality", "priority_score",
          "internet_exposed", "outside_working_hours"]]
        .rename(columns=lambda c: f"proposed_{c}" if c != "proposed_rank" else "proposed_rank_pos")
        .reset_index(drop=True)
    )
    comparison_top20 = pd.concat(
        [top20_base, top20_prop],
        axis=1,
    ).reset_index(drop=True)

    # ── 8. Flat metrics DataFrame for CSV export ──────────────────────────────
    metrics_df = pd.DataFrame([
        {"metric": k, "value": v} for k, v in metrics.items()
    ]).reset_index(drop=True)

    # ── 9. Figures ────────────────────────────────────────────────────────────
    fig_precision = _fig_precision(p10_base, p20_base, p10_prop, p20_prop)
    fig_elevation = _fig_elevation(
        avg_rank_crit_base, avg_rank_crit_prop,
        avg_rank_exp_base,  avg_rank_exp_prop,
        avg_rank_out_base,  avg_rank_out_prop,
        n=len(df),
    )
    fig_scatter   = _fig_scatter(rank_df)

    return EvalResult(
        metrics=metrics,
        baseline_ranked=df_baseline.reset_index(drop=True),
        proposed_ranked=df_proposed.reset_index(drop=True),
        comparison_top20=comparison_top20,
        metrics_df=metrics_df,
        fig_precision=fig_precision,
        fig_elevation=fig_elevation,
        fig_scatter=fig_scatter,
    )


# ── Figure builders ──────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply consistent thesis-quality styling to an axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def _fig_precision(
    p10_base: float, p20_base: float,
    p10_prop: float, p20_prop: float,
) -> plt.Figure:
    """Grouped bar chart: Precision@10 and Precision@20 for both approaches."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.array([0, 1])
    w = 0.32
    b1 = ax.bar(x - w / 2, [p10_base, p20_base], w,
                label="Baseline (severity-only)",
                color=BASELINE_COLOUR, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, [p10_prop, p20_prop], w,
                label="Proposed (XGBoost)",
                color=PROPOSED_COLOUR, edgecolor="white", linewidth=0.8)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision@10", "Precision@20"], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    _style_ax(
        ax,
        "Precision@K — Fraction of Top-K Alerts on High-Criticality Assets\n"
        "(asset criticality ≥ 4 out of 5)",
        "Precision",
    )
    fig.tight_layout()
    return fig


def _fig_elevation(
    crit_base: float, crit_prop: float,
    exp_base: float,  exp_prop: float,
    out_base: float,  out_prop: float,
    n: int,
) -> plt.Figure:
    """
    Grouped bar chart: average rank position for three context subgroups.
    Lower bar = ranked earlier in the list = better for triage.
    Includes a reference line at n/2 (random baseline).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Critical Assets\n(criticality=5)",
              "Internet-Exposed\nAssets",
              "Outside Working\nHours Alerts"]
    base_vals = [crit_base, exp_base, out_base]
    prop_vals = [crit_prop, exp_prop, out_prop]

    x = np.arange(len(labels))
    w = 0.32
    b1 = ax.bar(x - w / 2, base_vals, w,
                label="Baseline (severity-only)",
                color=BASELINE_COLOUR, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, prop_vals, w,
                label="Proposed (XGBoost)",
                color=PROPOSED_COLOUR, edgecolor="white", linewidth=0.8)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    # Reference line: expected average rank if ordering were random
    ax.axhline(n / 2, color=NEUTRAL_COLOUR, linestyle="--", linewidth=1.2,
               label=f"Random baseline (rank {n//2})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, n * 0.85)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    _style_ax(
        ax,
        "Average Rank Position of Context-Sensitive Alert Subgroups\n"
        "(lower rank = surfaced earlier = better triage)",
        "Average Rank Position",
    )

    # Annotate elevation arrows
    for i, (bv, pv) in enumerate(zip(base_vals, prop_vals)):
        if not (np.isnan(bv) or np.isnan(pv)) and abs(bv - pv) > 1:
            elev = bv - pv
            mid_x = x[i]
            ax.annotate(
                f"{elev:+.0f} positions",
                xy=(mid_x, pv + 2),
                xytext=(mid_x + 0.28, (bv + pv) / 2),
                fontsize=7.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
            )

    fig.tight_layout()
    return fig


def _fig_scatter(rank_df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot of baseline rank vs proposed rank for every alert.
    Points above the diagonal (y > x) moved UP in the proposed ranking.
    Colour-coded by asset_criticality.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    crit = rank_df["asset_criticality"].values
    cmap = plt.cm.RdYlGn   # red (low) → green (high) criticality

    sc = ax.scatter(
        rank_df["baseline_rank"],
        rank_df["proposed_rank"],
        c=crit, cmap=cmap, vmin=1, vmax=5,
        s=28, alpha=0.75, edgecolors="none",
    )

    # Diagonal reference line (equal ranking)
    lim = len(rank_df) + 2
    ax.plot([1, lim], [1, lim], color=NEUTRAL_COLOUR,
            linestyle="--", linewidth=1.2, label="Equal rank (diagonal)")

    # Shade regions
    ax.fill_between([1, lim], [1, lim], [lim, lim],
                    alpha=0.05, color=PROPOSED_COLOUR,
                    label="Proposed ranked higher (above diagonal)")
    ax.fill_between([1, lim], [1, 1], [1, lim],
                    alpha=0.05, color=BASELINE_COLOUR,
                    label="Baseline ranked higher (below diagonal)")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Asset Criticality (1–5)", fontsize=9)

    ax.set_xlim(1, lim)
    ax.set_ylim(1, lim)
    ax.set_xlabel("Baseline Rank (severity-only)", fontsize=10)
    ax.set_ylabel("Proposed Rank (XGBoost)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _style_ax(
        ax,
        "Rank Divergence: Baseline vs. Proposed\n"
        "(each point = one alert; colour = asset criticality)",
        "Proposed Rank",
    )
    fig.tight_layout()
    return fig


# ── CSV export builder ───────────────────────────────────────────────────────

def build_export_zip(result: EvalResult) -> bytes:
    """
    Package the evaluation results into a ZIP of four CSV files:
        metrics.csv           — flat key/value metrics table
        ranking_baseline.csv  — all alerts in baseline order with rank
        ranking_proposed.csv  — all alerts in proposed order with rank
        comparison_top20.csv  — top-20 side-by-side comparison
    Returns the ZIP as bytes for st.download_button.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # metrics
        zf.writestr("metrics.csv", result.metrics_df.to_csv(index=False))

        # baseline ranking (select key columns only)
        base_cols = [
            "baseline_rank", "alert_id", "timestamp", "alert_category",
            "severity", "affected_asset", "asset_criticality",
            "internet_exposed", "outside_working_hours",
            "user", "user_role", "source_ip",
        ]
        base_cols = [c for c in base_cols if c in result.baseline_ranked.columns]
        zf.writestr(
            "ranking_baseline.csv",
            result.baseline_ranked[base_cols].to_csv(index=False),
        )

        # proposed ranking
        prop_cols = [
            "proposed_rank", "alert_id", "timestamp", "alert_category",
            "severity", "affected_asset", "asset_criticality",
            "internet_exposed", "outside_working_hours",
            "priority_label", "priority_score",
            "user", "user_role", "source_ip",
        ]
        prop_cols = [c for c in prop_cols if c in result.proposed_ranked.columns]
        zf.writestr(
            "ranking_proposed.csv",
            result.proposed_ranked[prop_cols].to_csv(index=False),
        )

        # top-20 comparison
        zf.writestr(
            "comparison_top20.csv",
            result.comparison_top20.to_csv(index=False),
        )

    buf.seek(0)
    return buf.read()


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.generate_alerts import generate
    from modules.context_enricher import enrich
    from modules.model import load_model, predict
    from modules.explainer import build_explainer, explain_dataframe

    raw      = generate(n=200, out_path="data/alerts_raw.csv")
    enriched = enrich(raw)
    model    = load_model()
    explainer= build_explainer(model)
    scored   = predict(enriched, model)
    explained= explain_dataframe(explainer, scored)

    result = evaluate(explained)

    print("=" * 55)
    print("EVALUATION METRICS")
    print("=" * 55)
    for k, v in result.metrics.items():
        print(f"  {k:<40s}  {v}")

    print("\nTop-5 Baseline vs Proposed:")
    print(result.comparison_top20.head(5).to_string())
