"""
RiskShield AI — Financial Fraud Detection Dashboard
Clean, readable UI with clear sections and visual hierarchy.
"""

import json
import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
import streamlit as st
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TARGET_COL  # type: ignore


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

RISK_LEVELS = {
    "CRITICAL": {"min": 0.85, "color": "#DC2626", "bg": "#FEF2F2", "border": "#FECACA"},
    "HIGH":     {"min": 0.65, "color": "#D97706", "bg": "#FFFBEB", "border": "#FDE68A"},
    "MEDIUM":   {"min": 0.40, "color": "#2563EB", "bg": "#EFF6FF", "border": "#BFDBFE"},
    "LOW":      {"min": 0.00, "color": "#16A34A", "bg": "#F0FDF4", "border": "#BBF7D0"},
}


# ─────────────────────────────────────────────
#  RISK HELPERS
# ─────────────────────────────────────────────

def assign_risk_level(prob: float) -> str:
    if prob >= 0.85: return "CRITICAL"
    if prob >= 0.65: return "HIGH"
    if prob >= 0.40: return "MEDIUM"
    return "LOW"


def compute_anomaly_score(df: pd.DataFrame) -> pd.Series:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = [TARGET_COL, "fraud_probability", "fraud_flag", "risk_score"]
    num_cols = [c for c in num_cols if c not in drop][:6]
    if not num_cols:
        return pd.Series(0.0, index=df.index)
    zscores = df[num_cols].apply(lambda col: ((col - col.mean()) / (col.std() + 1e-9)).abs())
    return zscores.mean(axis=1)


def score_transactions(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df_feat = df.drop(columns=[TARGET_COL], errors="ignore")
    probs   = model.predict_proba(df_feat)[:, 1]
    flags   = (probs >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = probs
    out["fraud_flag"]        = flags
    out["risk_level"]        = [assign_risk_level(p) for p in probs]

    anomaly          = compute_anomaly_score(out)
    out["risk_score"] = (0.75 * probs + 0.25 * (anomaly / (anomaly.max() + 1e-9))).round(4)
    return out


# ─────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    import platform
    path = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
    if not path.exists():
        st.error("Model file not found. Run `python -m src.train_model` first.")
        st.stop()
    try:
        return joblib.load(path)
    except AttributeError as e:
        st.error(
            f"Model cannot be loaded — Python version mismatch "
            f"(running {platform.python_version()}). "
            "Retrain on Python 3.11 and redeploy."
        )
        st.stop()


@st.cache_data
def load_threshold():
    p = PROJECT_ROOT / "models" / "threshold.json"
    if not p.exists():
        return {"threshold": 0.5}
    return json.loads(p.read_text())


@st.cache_data
def load_sample_data():
    p = PROCESSED_DATA_DIR / "transactions_test.csv"
    if not p.exists():
        st.error("Test data not found. Run `python -m src.data_prep` first.")
        st.stop()
    return pd.read_csv(p)


# ─────────────────────────────────────────────
#  SHAP
# ─────────────────────────────────────────────

@st.cache_resource
def get_shap_explainer(_model):
    pre   = _model.named_steps["preprocess"]
    clf   = _model.named_steps["clf"]
    return shap.TreeExplainer(clf), pre, pre.get_feature_names_out()


def shap_bar_chart(shap_vals, feature_names, max_f=12):
    sv = np.asarray(shap_vals).reshape(-1)
    fn = np.asarray(feature_names)
    n  = min(len(sv), len(fn))
    sv, fn = sv[:n], fn[:n]
    idx    = np.argsort(np.abs(sv))[::-1][:max_f]
    sv, fn = sv[idx], fn[idx]

    fig, ax = plt.subplots(figsize=(8, max(3, len(fn) * 0.45)))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F9FAFB")

    colors = ["#DC2626" if v > 0 else "#2563EB" for v in sv]
    ax.barh(range(len(fn)), sv, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(fn)))
    ax.set_yticklabels(fn, fontsize=9, color="#374151")
    ax.invert_yaxis()
    ax.axvline(0, color="#D1D5DB", linewidth=0.8)
    ax.set_xlabel("SHAP value  (positive = more fraud risk)", fontsize=9, color="#6B7280")
    ax.tick_params(axis="x", colors="#9CA3AF", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#E5E7EB")

    red_p  = mpatches.Patch(color="#DC2626", label="Increases fraud risk")
    blue_p = mpatches.Patch(color="#2563EB", label="Decreases fraud risk")
    ax.legend(handles=[red_p, blue_p], fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor="#E5E7EB")
    plt.tight_layout()
    return fig


def explain_row(model, df_scored, row_idx):
    explainer, pre, feat_names = get_shap_explainer(model)
    drop = [TARGET_COL, "fraud_probability", "fraud_flag", "risk_level", "risk_score"]
    feat_df = df_scored.drop(columns=[c for c in drop if c in df_scored.columns])
    x       = feat_df.iloc[[row_idx]]
    xt      = pre.transform(x)

    try:
        import scipy.sparse as sp
        xt = xt.toarray() if sp.issparse(xt) else xt
    except ImportError:
        pass

    sv = explainer.shap_values(xt)
    sv = sv[1][0] if isinstance(sv, list) else sv[0]
    return shap_bar_chart(sv, feat_names)


# ─────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────

def _base_fig(w=8, h=3):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F9FAFB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.tick_params(colors="#9CA3AF", labelsize=8)
    return fig, ax


def chart_histogram(df_scored, thr):
    fig, ax = _base_fig(8, 3)
    probs = df_scored["fraud_probability"]
    ax.hist(probs[probs < thr],  bins=35, color="#93C5FD", alpha=0.85, label="Below threshold")
    ax.hist(probs[probs >= thr], bins=35, color="#FCA5A5", alpha=0.85, label="Flagged")
    ax.axvline(thr,  color="#1D4ED8", linewidth=1.5, linestyle="--", label=f"Threshold ({thr:.2f})")
    ax.axvline(0.85, color="#DC2626", linewidth=1,   linestyle=":",  label="Critical (0.85)")
    ax.set_xlabel("Fraud probability", fontsize=9, color="#6B7280")
    ax.set_ylabel("Number of transactions", fontsize=9, color="#6B7280")
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#E5E7EB")
    plt.tight_layout()
    return fig


def chart_timeline(df_scored):
    fig, ax = _base_fig(8, 2.8)
    probs = df_scored["fraud_probability"].values
    x     = np.arange(len(probs))
    ax.fill_between(x, probs, color="#BFDBFE", alpha=0.5)
    ax.plot(x, probs, color="#3B82F6", linewidth=0.8)

    crit = df_scored["risk_level"] == "CRITICAL"
    if crit.any():
        ax.scatter(x[crit.values], probs[crit.values],
                   color="#DC2626", s=18, zorder=5, label="Critical")

    ax.axhline(0.85, color="#DC2626", linewidth=0.8, linestyle="--", alpha=0.7, label="Critical (0.85)")
    ax.axhline(0.65, color="#D97706", linewidth=0.8, linestyle="--", alpha=0.7, label="High (0.65)")
    ax.set_xlabel("Transaction index", fontsize=9, color="#6B7280")
    ax.set_ylabel("Fraud probability", fontsize=9, color="#6B7280")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#E5E7EB")
    plt.tight_layout()
    return fig


def chart_risk_bars(level_counts):
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = [level_counts.get(l, 0) for l in levels]
    colors = [RISK_LEVELS[l]["color"] for l in levels]

    fig, ax = _base_fig(4, 2.8)
    bars = ax.bar(levels, counts, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha="center", va="bottom", fontsize=9, color="#374151", fontweight="500")
    ax.set_ylabel("Transactions", fontsize=9, color="#6B7280")
    ax.set_ylim(0, max(counts) * 1.2 + 1)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  PAGE CSS
# ─────────────────────────────────────────────

PAGE_CSS = """
<style>
.stApp { background: #F8FAFC; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px;
}
.kpi-label { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #111827; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #9CA3AF; margin-top: 6px; }

.alert {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;
    border-left: 4px solid;
}
.alert-title { font-size: 14px; font-weight: 600; margin-bottom: 2px; }
.alert-body  { font-size: 13px; }
.alert-critical { background:#FEF2F2; border-color:#DC2626; color:#7F1D1D; }
.alert-high     { background:#FFFBEB; border-color:#D97706; color:#78350F; }
.alert-clear    { background:#F0FDF4; border-color:#16A34A; color:#14532D; }

.section-title {
    font-size: 12px; font-weight: 600; color: #374151;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 0 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid #E5E7EB;
}

.detail-card {
    background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 20px;
}
</style>
"""


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="RiskShield AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    model     = load_model()
    thr_info  = load_threshold()
    default_t = float(thr_info.get("threshold", 0.5))

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.markdown("## 🛡️ RiskShield AI")
        st.caption("Financial fraud detection engine")
        st.divider()

        st.markdown("#### Data source")
        uploaded = st.file_uploader(
            "Upload transaction CSV", type=["csv"],
            help="Leave empty to use the built-in test dataset"
        )
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"{len(df_raw):,} rows loaded")
        else:
            df_raw = load_sample_data()
            st.info("Using built-in test dataset")

        st.divider()
        st.markdown("#### Detection threshold")
        thr = st.slider(
            "Flag if probability ≥",
            min_value=0.0, max_value=1.0, value=default_t, step=0.01,
        )
        st.caption(f"Current threshold: **{thr:.2f}**")
        st.caption("Lower value = more sensitive (more flags)")

        st.divider()
        st.markdown("#### Risk level guide")
        for name, cfg in RISK_LEVELS.items():
            st.markdown(
                f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
                f'font-size:11px;font-weight:700;background:{cfg["bg"]};'
                f'color:{cfg["color"]};border:1px solid {cfg["border"]};">{name}</span>'
                f'&nbsp;&nbsp;<span style="font-size:12px;color:#6B7280;">≥ {cfg["min"]:.0%}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    # ── Score data ────────────────────────────
    df_scored    = score_transactions(model, df_raw, thr)
    level_counts = df_scored["risk_level"].value_counts().to_dict()
    n_critical   = level_counts.get("CRITICAL", 0)
    n_high       = level_counts.get("HIGH", 0)
    n_flagged    = int(df_scored["fraud_flag"].sum())
    n_total      = len(df_scored)

    # ── Page header ───────────────────────────
    st.markdown("# 🛡️ RiskShield AI")
    st.markdown(
        '<p style="color:#6B7280;margin-top:-10px;margin-bottom:24px;font-size:15px;">'
        "Real-time transaction monitoring &nbsp;·&nbsp; Behavioral risk scoring &nbsp;·&nbsp; Explainable AI"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Alert banner ──────────────────────────
    if n_critical > 0:
        st.markdown(
            f'<div class="alert alert-critical">'
            f'<div><div class="alert-title">⚠ Critical risk detected</div>'
            f'<div class="alert-body">{n_critical} transaction(s) scored above 0.85 — '
            f'immediate review recommended.</div></div></div>',
            unsafe_allow_html=True,
        )
    elif n_high > 0:
        st.markdown(
            f'<div class="alert alert-high">'
            f'<div><div class="alert-title">! High-risk transactions present</div>'
            f'<div class="alert-body">{n_high} transaction(s) scored 0.65–0.84. Review advised.</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert alert-clear">'
            '<div><div class="alert-title">✓ All clear</div>'
            '<div class="alert-body">No critical or high-risk transactions detected in this dataset.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── KPI cards ─────────────────────────────
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Total transactions</div>
            <div class="kpi-value">{n_total:,}</div>
            <div class="kpi-sub">In current dataset</div>
          </div>
          <div class="kpi-card" style="border-top:3px solid #DC2626;">
            <div class="kpi-label">Critical risk</div>
            <div class="kpi-value" style="color:#DC2626;">{n_critical:,}</div>
            <div class="kpi-sub">Probability ≥ 0.85</div>
          </div>
          <div class="kpi-card" style="border-top:3px solid #D97706;">
            <div class="kpi-label">High risk</div>
            <div class="kpi-value" style="color:#D97706;">{n_high:,}</div>
            <div class="kpi-sub">Probability 0.65 – 0.84</div>
          </div>
          <div class="kpi-card" style="border-top:3px solid #2563EB;">
            <div class="kpi-label">Flagged by model</div>
            <div class="kpi-value" style="color:#2563EB;">{n_flagged:,}</div>
            <div class="kpi-sub">{n_flagged / n_total * 100:.1f}% of all transactions</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 1: Risk distribution charts ───
    st.markdown('<p class="section-title">Risk distribution</p>', unsafe_allow_html=True)

    col_tl, col_bar = st.columns([3, 1], gap="medium")
    with col_tl:
        st.caption("Fraud probability across all transactions — red dots mark critical-risk transactions")
        fig = chart_timeline(df_scored)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_bar:
        st.caption("Count by risk level")
        fig = chart_risk_bars(level_counts)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Section 2: Probability histogram ──────
    st.markdown(
        '<p class="section-title" style="margin-top:28px;">Probability distribution</p>',
        unsafe_allow_html=True,
    )
    st.caption("Blue = safe (below threshold)  ·  Red = flagged  ·  Dashed blue line = current threshold")
    fig = chart_histogram(df_scored, thr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Section 3: Transaction tables ─────────
    st.markdown(
        '<p class="section-title" style="margin-top:28px;">Suspicious transactions</p>',
        unsafe_allow_html=True,
    )

    def prep_display(df):
        priority = ["fraud_probability", "risk_score", "risk_level", "fraud_flag"]
        if TARGET_COL in df.columns:
            priority.append(TARGET_COL)
        other = [c for c in df.columns if c not in priority]
        return df[other[:7] + [c for c in priority if c in df.columns]]

    tab_crit, tab_all, tab_export = st.tabs([
        f"Critical & High ({n_critical + n_high} transactions)",
        "All transactions",
        "Export",
    ])

    with tab_crit:
        suspicious = df_scored[df_scored["risk_level"].isin(["CRITICAL", "HIGH"])].sort_values(
            "fraud_probability", ascending=False
        )
        if suspicious.empty:
            st.success("No critical or high-risk transactions found.")
        else:
            st.caption(f"Showing {len(suspicious)} transactions with risk level CRITICAL or HIGH, sorted by fraud probability.")
            st.dataframe(
                prep_display(suspicious).style.background_gradient(
                    subset=["fraud_probability"], cmap="Reds", vmin=0, vmax=1
                ),
                use_container_width=True, height=380,
            )

    with tab_all:
        c1, c2 = st.columns([1, 2])
        with c1:
            n_show = st.slider("Rows to display", 20, min(500, n_total), 100, step=10)
        with c2:
            sort_options = ["fraud_probability", "risk_score"] + [
                c for c in df_scored.columns
                if c not in ["fraud_probability", "risk_score", "fraud_flag", "risk_level", TARGET_COL]
            ]
            sort_col = st.selectbox("Sort by", options=sort_options)

        df_view = df_scored.sort_values(sort_col, ascending=False).head(n_show)
        st.dataframe(
            prep_display(df_view).style.background_gradient(
                subset=["fraud_probability"], cmap="RdYlGn_r", vmin=0, vmax=1
            ),
            use_container_width=True, height=420,
        )

    with tab_export:
        export = df_scored[df_scored["fraud_flag"] == 1].sort_values("fraud_probability", ascending=False)
        st.markdown(f"**{len(export):,} flagged transactions** are ready for download.")
        st.caption("The export includes fraud probability, risk score, risk level, and all original columns.")
        st.download_button(
            label=f"Download flagged transactions — {datetime.date.today()}.csv",
            data=export.to_csv(index=False).encode(),
            file_name=f"flagged_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    # ── Section 4: Transaction explainer ──────
    st.markdown(
        '<p class="section-title" style="margin-top:32px;">Explain a single transaction</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Select any transaction to see a breakdown of which features made the model flag it as fraud."
    )

    id_col = next((c for c in ["transaction_id", "id", "txn_id"] if c in df_scored.columns), None)

    left, right = st.columns([1, 2], gap="large")

    with left:
        if id_col:
            sel_id  = st.selectbox(f"Select transaction by {id_col}", df_scored[id_col].tolist())
            row_idx = df_scored.index[df_scored[id_col] == sel_id][0]
        else:
            row_idx = st.number_input("Row index", min_value=0, max_value=n_total - 1, value=0)

        row     = df_scored.iloc[row_idx]
        rl      = row.get("risk_level", "LOW")
        cfg     = RISK_LEVELS[rl]
        prob    = float(row.get("fraud_probability", 0))
        rs      = float(row.get("risk_score", 0))
        flagged = bool(row.get("fraud_flag", 0))

        st.markdown(
            f"""
            <div class="detail-card" style="border-top:4px solid {cfg['color']};margin-top:12px;">
              <div style="font-size:40px;font-weight:700;color:{cfg['color']};line-height:1;">{prob:.1%}</div>
              <div style="font-size:12px;font-weight:700;letter-spacing:1px;color:{cfg['color']};margin-top:4px;">{rl} RISK</div>
              <hr style="border:none;border-top:1px solid #E5E7EB;margin:14px 0;">
              <table style="width:100%;font-size:13px;border-collapse:collapse;">
                <tr>
                  <td style="color:#6B7280;padding:5px 0;">Fraud probability</td>
                  <td style="text-align:right;font-weight:600;color:#111827;">{prob:.4f}</td>
                </tr>
                <tr>
                  <td style="color:#6B7280;padding:5px 0;">Composite risk score</td>
                  <td style="text-align:right;font-weight:600;color:#111827;">{rs:.4f}</td>
                </tr>
                <tr>
                  <td style="color:#6B7280;padding:5px 0;">Model decision</td>
                  <td style="text-align:right;font-weight:700;color:{'#DC2626' if flagged else '#16A34A'};">
                    {'FLAGGED' if flagged else 'CLEAR'}
                  </td>
                </tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Raw feature values for this transaction**")
        drop_display = ["fraud_probability", "risk_score", "fraud_flag", "risk_level"]
        display_row  = row.drop(labels=[c for c in drop_display if c in row.index], errors="ignore")
        st.dataframe(display_row.to_frame().T, use_container_width=True)

    st.markdown("**What drove this prediction?**")
    st.caption(
        "Each bar shows one feature's contribution. "
        "Red = pushed the score toward fraud. "
        "Blue = pushed it away from fraud. "
        "Longer bar = stronger influence."
    )

    with st.spinner("Computing SHAP values…"):
        try:
            fig_shap = explain_row(model, df_scored, row_idx)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable for this row: {e}")

    # ── Footer ────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="text-align:center;font-size:12px;color:#9CA3AF;margin:0;">'
        "RiskShield AI &nbsp;·&nbsp; Random Forest + SHAP &nbsp;·&nbsp; "
        "Built for real-time fintech fraud detection"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
