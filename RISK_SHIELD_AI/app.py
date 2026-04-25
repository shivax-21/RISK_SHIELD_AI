import json
from pathlib import Path
import time
import datetime

import joblib
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


# ------------- Risk Scoring Helpers ------------------

RISK_THRESHOLDS = {
    "CRITICAL": 0.85,
    "HIGH":     0.65,
    "MEDIUM":   0.40,
    "LOW":      0.00,
}

RISK_COLORS = {
    "CRITICAL": "#FF2D2D",
    "HIGH":     "#FF8C00",
    "MEDIUM":   "#FFD700",
    "LOW":      "#00C853",
}

def assign_risk_level(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif prob >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif prob >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "LOW"

def compute_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dynamic behavioral risk signals on top of raw features.
    Works with whatever numeric columns exist.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [TARGET_COL, "fraud_probability", "fraud_flag", "risk_level", "risk_score"]
    num_cols = [c for c in num_cols if c not in exclude]

    if len(num_cols) >= 1:
        # Z-score based anomaly signal: how many std devs away from mean
        for col in num_cols[:5]:  # limit to first 5 numeric cols
            col_mean = df[col].mean()
            col_std  = df[col].std() + 1e-9
            df[f"_zscore_{col}"] = ((df[col] - col_mean) / col_std).abs()

        zscore_cols = [c for c in df.columns if c.startswith("_zscore_")]
        df["_anomaly_score"] = df[zscore_cols].mean(axis=1)
    else:
        df["_anomaly_score"] = 0.0

    return df

def dynamic_risk_score(fraud_prob: float, anomaly_score: float) -> float:
    """
    Blend model probability with behavioral anomaly signal.
    Gives a richer 0-100 risk score.
    """
    blended = 0.75 * fraud_prob + 0.25 * min(anomaly_score / 5.0, 1.0)
    return round(blended * 100, 1)


# ------------- Model / Data Loaders ------------------

@st.cache_resource
def load_model():
    import platform
    model_path = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
    if not model_path.exists():
        st.error(
            f"Model not found at {model_path}. "
            "Run `python -m src.train_model` first."
        )
        st.stop()
    try:
        model = joblib.load(model_path)
        return model
    except AttributeError as e:
        st.error(
            f"**Model load failed** — Python version mismatch.\n\n"
            f"Current Python: `{platform.python_version()}` — Error: `{e}`\n\n"
            "Retrain the model with the same Python version as this deployment."
        )
        st.stop()

@st.cache_data
def load_threshold():
    threshold_path = PROJECT_ROOT / "models" / "threshold.json"
    if not threshold_path.exists():
        return {"threshold": 0.5}
    with threshold_path.open() as f:
        return json.load(f)

@st.cache_data
def load_sample_data():
    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
    if not test_path.exists():
        st.error(
            f"Processed test data not found at {test_path}. "
            "Run `python -m src.data_prep` first."
        )
        st.stop()
    return pd.read_csv(test_path)


# ------------- Scoring ------------------

def score_transactions(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df_features = df.drop(columns=[TARGET_COL], errors="ignore")
    probs  = model.predict_proba(df_features)[:, 1]
    flags  = (probs >= threshold).astype(int)

    df_scored = df.copy()
    df_scored["fraud_probability"] = probs
    df_scored["fraud_flag"]        = flags

    # Behavioral signals
    df_scored = compute_behavior_features(df_scored)
    anomaly   = df_scored["_anomaly_score"].values

    df_scored["risk_score"] = [
        dynamic_risk_score(p, a) for p, a in zip(probs, anomaly)
    ]
    df_scored["risk_level"] = df_scored["fraud_probability"].apply(assign_risk_level)

    # Drop internal cols
    df_scored = df_scored.drop(columns=[c for c in df_scored.columns if c.startswith("_zscore_")])
    return df_scored


# ------------- SHAP ------------------

@st.cache_resource
def get_shap_explainer(_model):
    preprocessor = _model.named_steps["preprocess"]
    clf          = _model.named_steps["clf"]
    explainer    = shap.TreeExplainer(clf)
    feature_names = preprocessor.get_feature_names_out()
    return explainer, preprocessor, feature_names

def plot_single_shap_bar(shap_values, feature_names, max_features=10):
    shap_values  = np.asarray(shap_values).reshape(-1)
    feature_names = np.asarray(feature_names)
    n = min(len(shap_values), len(feature_names))
    shap_values, feature_names = shap_values[:n], feature_names[:n]
    idx = np.argsort(np.abs(shap_values))[::-1][:max_features]
    sv, fn = shap_values[idx], feature_names[idx]

    colors = ["#FF4444" if v > 0 else "#4488FF" for v in sv]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.barh(np.arange(len(fn)), sv, color=colors)
    ax.set_yticks(np.arange(len(fn)))
    ax.set_yticklabels(fn, color="white", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on fraud probability)", color="white")
    ax.set_title("Feature contributions for this transaction", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    red_patch  = mpatches.Patch(color="#FF4444", label="Increases fraud risk")
    blue_patch = mpatches.Patch(color="#4488FF", label="Decreases fraud risk")
    ax.legend(handles=[red_patch, blue_patch], facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    plt.tight_layout()
    return fig

def explain_single_transaction(model, df_scored, row_idx):
    explainer, preprocessor, feature_names = get_shap_explainer(model)
    cols_to_drop = [TARGET_COL, "fraud_probability", "fraud_flag", "risk_level", "risk_score", "_anomaly_score"]
    features_df  = df_scored.drop(columns=[c for c in cols_to_drop if c in df_scored.columns])
    x_row        = features_df.iloc[[row_idx]]
    x_transformed = preprocessor.transform(x_row)
    try:
        import scipy.sparse as sp
        x_for_shap = x_transformed.toarray() if sp.issparse(x_transformed) else x_transformed
    except ImportError:
        x_for_shap = x_transformed

    shap_vals = explainer.shap_values(x_for_shap)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]
    return plot_single_shap_bar(sv, feature_names)


# ------------- UI Helpers ------------------

def risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, "#888")
    return f'<span style="background:{color};color:#000;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:700;">{level}</span>'

def render_alert_banner(critical_count: int, high_count: int):
    if critical_count > 0:
        st.markdown(
            f"""
            <div style="background:linear-gradient(90deg,#FF2D2D22,#FF2D2D44);
                        border-left:4px solid #FF2D2D;padding:12px 20px;border-radius:8px;margin-bottom:16px;">
                <b style="color:#FF2D2D;font-size:16px;">🚨 ALERT:</b>
                <span style="color:#fff;"> {critical_count} CRITICAL risk transaction(s) detected — immediate review required!</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif high_count > 0:
        st.markdown(
            f"""
            <div style="background:linear-gradient(90deg,#FF8C0022,#FF8C0044);
                        border-left:4px solid #FF8C00;padding:12px 20px;border-radius:8px;margin-bottom:16px;">
                <b style="color:#FF8C00;font-size:16px;">⚠️ WARNING:</b>
                <span style="color:#fff;"> {high_count} HIGH risk transaction(s) flagged for review.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:linear-gradient(90deg,#00C85322,#00C85344);
                        border-left:4px solid #00C853;padding:12px 20px;border-radius:8px;margin-bottom:16px;">
                <b style="color:#00C853;font-size:16px;">✅ CLEAR:</b>
                <span style="color:#fff;"> No critical or high-risk transactions detected.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

def plot_risk_donut(level_counts: dict):
    levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sizes  = [level_counts.get(l, 0) for l in levels]
    colors = [RISK_COLORS[l] for l in levels]
    non_zero = [(s, c, l) for s, c, l in zip(sizes, colors, levels) if s > 0]
    if not non_zero:
        return None
    sizes_nz, colors_nz, labels_nz = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    wedges, _ = ax.pie(
        sizes_nz, colors=colors_nz, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="#0E1117", linewidth=2),
    )
    ax.legend(
        wedges, [f"{l}: {s}" for l, s in zip(labels_nz, sizes_nz)],
        loc="lower center", ncol=2, facecolor="#1a1a1a",
        labelcolor="white", fontsize=9, framealpha=0.8,
    )
    ax.set_title("Risk Distribution", color="white", fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

def plot_risk_timeline(df_scored: pd.DataFrame):
    """Show fraud probability across transaction index as a timeline."""
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    probs = df_scored["fraud_probability"].values
    idx   = np.arange(len(probs))

    ax.fill_between(idx, probs, alpha=0.15, color="#FF4444")
    ax.plot(idx, probs, color="#FF4444", linewidth=0.8, alpha=0.7)

    # Mark critical points
    crit_mask = df_scored["risk_level"] == "CRITICAL"
    if crit_mask.any():
        ax.scatter(idx[crit_mask], probs[crit_mask], color="#FF2D2D", s=20, zorder=5, label="Critical")

    ax.axhline(0.65, color="#FF8C00", linestyle="--", linewidth=0.8, alpha=0.6, label="High threshold")
    ax.axhline(0.85, color="#FF2D2D", linestyle="--", linewidth=0.8, alpha=0.6, label="Critical threshold")
    ax.set_xlabel("Transaction Index", color="white", fontsize=9)
    ax.set_ylabel("Fraud Probability", color="white", fontsize=9)
    ax.set_title("Real-Time Risk Timeline", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    plt.tight_layout()
    return fig

def style_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return display-ready dataframe with key columns."""
    display_cols = ["fraud_probability", "risk_score", "risk_level", "fraud_flag"]
    available    = [c for c in display_cols if c in df.columns]
    other_cols   = [c for c in df.columns if c not in display_cols + [TARGET_COL, "_anomaly_score"]]
    return df[other_cols[:6] + available].copy()


# ------------- Main App ------------------

def main():
    st.set_page_config(
        page_title="RiskShield AI — Fraud Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Custom CSS ----
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: #0E1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .metric-card .delta {
        font-size: 0.85rem;
        margin-top: 6px;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        border-bottom: 2px solid #2a2a4a;
        padding-bottom: 8px;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .suspicious-row { background: #FF2D2D15 !important; }

    div[data-testid="stSidebar"] {
        background: #0a0a1a;
        border-right: 1px solid #1a1a3a;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    col_logo, col_title = st.columns([1, 10])
    with col_logo:
        st.markdown("<div style='font-size:2.5rem;padding-top:8px'>🛡️</div>", unsafe_allow_html=True)
    with col_title:
        st.markdown("""
        <h1 style='margin:0;font-size:1.8rem;font-weight:700;color:#fff;'>
            RiskShield AI
            <span style='font-size:0.85rem;color:#8888aa;font-weight:400;margin-left:12px;'>
                Real-Time Fraud Detection Engine
            </span>
        </h1>
        <p style='color:#8888aa;font-size:0.85rem;margin:0;'>
            Behavioral analysis · Dynamic risk scoring · Explainable AI
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Load model ----
    model          = load_model()
    threshold_info = load_threshold()
    default_thr    = float(threshold_info.get("threshold", 0.5))

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        st.markdown("---")

        st.markdown("**Decision Threshold**")
        thr = st.slider(
            "Flag as fraud if probability ≥",
            min_value=0.0, max_value=1.0,
            value=default_thr, step=0.01,
        )
        st.caption(f"Currently: `{thr:.2f}` — Lower = more sensitive")

        st.markdown("---")
        st.markdown("**Risk Level Filter**")
        show_levels = st.multiselect(
            "Show transactions with risk level:",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )

        st.markdown("---")
        st.markdown("**Data Source**")
        uploaded_file = st.file_uploader("Upload CSV with transactions", type=["csv"])

        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_raw):,} rows from upload")
        else:
            df_raw = load_sample_data()
            st.info("Using built-in test dataset")

        st.markdown("---")
        st.markdown("**Risk Thresholds**")
        for level, color in RISK_COLORS.items():
            thr_val = RISK_THRESHOLDS[level]
            st.markdown(
                f'<span style="color:{color};font-weight:700;">{level}</span>'
                f'<span style="color:#888;font-size:0.8rem;"> ≥ {thr_val}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto-refresh simulation", value=False)
        if auto_refresh:
            st.caption("Simulating real-time feed...")

    # ---- Score ----
    df_scored = score_transactions(model, df_raw, thr)

    # Apply filter
    df_filtered = df_scored[df_scored["risk_level"].isin(show_levels)] if show_levels else df_scored

    # ---- Counts ----
    level_counts  = df_scored["risk_level"].value_counts().to_dict()
    critical_count = level_counts.get("CRITICAL", 0)
    high_count     = level_counts.get("HIGH", 0)

    # ---- Alert Banner ----
    render_alert_banner(critical_count, high_count)

    # ---- KPI Row ----
    st.markdown('<div class="section-header">📊 Live Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    total = len(df_scored)
    flagged = df_scored["fraud_flag"].sum()
    avg_risk = df_scored["risk_score"].mean()
    true_rate = df_scored[TARGET_COL].mean() * 100 if TARGET_COL in df_scored.columns else None

    with k1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{total:,}</div>
            <div class="label">Total Transactions</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#FF2D2D;">{critical_count}</div>
            <div class="label">Critical Risk</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#FF8C00;">{high_count}</div>
            <div class="label">High Risk</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#FFD700;">{flagged:,}</div>
            <div class="label">Flagged by Model</div>
            <div class="delta" style="color:#888;">{flagged/total*100:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#{'FF4444' if avg_risk > 50 else '00C853'};">{avg_risk:.1f}</div>
            <div class="label">Avg Risk Score</div>
            <div class="delta" style="color:#888;">0–100 scale</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Charts Row ----
    st.markdown('<div class="section-header">📈 Risk Analysis</div>', unsafe_allow_html=True)
    chart1, chart2 = st.columns([3, 2])

    with chart1:
        fig_timeline = plot_risk_timeline(df_scored)
        st.pyplot(fig_timeline, use_container_width=True)
        plt.close()

    with chart2:
        fig_donut = plot_risk_donut(level_counts)
        if fig_donut:
            st.pyplot(fig_donut, use_container_width=True)
            plt.close()

    # ---- Probability Histogram ----
    st.markdown('<div class="section-header">📉 Probability Distribution</div>', unsafe_allow_html=True)
    fig_hist, ax_hist = plt.subplots(figsize=(10, 3))
    fig_hist.patch.set_facecolor("#0E1117")
    ax_hist.set_facecolor("#0E1117")
    ax_hist.hist(df_scored["fraud_probability"], bins=40, color="#4488FF", alpha=0.7, edgecolor="#0E1117")
    ax_hist.axvline(thr,  color="#FFD700",  linestyle="--", linewidth=1.5, label=f"Threshold ({thr:.2f})")
    ax_hist.axvline(0.65, color="#FF8C00",  linestyle=":",  linewidth=1,   label="High (0.65)")
    ax_hist.axvline(0.85, color="#FF2D2D",  linestyle=":",  linewidth=1,   label="Critical (0.85)")
    ax_hist.set_xlabel("Fraud Probability", color="white")
    ax_hist.set_ylabel("Count", color="white")
    ax_hist.tick_params(colors="white")
    ax_hist.spines[:].set_color("#333")
    ax_hist.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
    st.pyplot(fig_hist, use_container_width=True)
    plt.close()

    # ---- Suspicious Transactions Table ----
    st.markdown('<div class="section-header">🚨 Suspicious Transactions</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔴 Critical & High Risk", "📋 All Filtered Transactions", "📥 Export"])

    with tab1:
        suspicious = df_scored[df_scored["risk_level"].isin(["CRITICAL", "HIGH"])].sort_values(
            "risk_score", ascending=False
        )
        if len(suspicious) == 0:
            st.success("✅ No critical or high-risk transactions found.")
        else:
            st.markdown(f"**{len(suspicious)} suspicious transaction(s) requiring attention:**")
            display_df = style_risk_table(suspicious)
            st.dataframe(
                display_df.style.background_gradient(
                    subset=["fraud_probability", "risk_score"],
                    cmap="RdYlGn_r"
                ),
                use_container_width=True,
                height=350,
            )

    with tab2:
        top_n = st.slider("Show top N by risk score", 10, min(500, len(df_filtered)), 50)
        display_all = style_risk_table(
            df_filtered.sort_values("risk_score", ascending=False).head(top_n)
        )
        st.dataframe(
            display_all.style.background_gradient(
                subset=["fraud_probability", "risk_score"],
                cmap="RdYlGn_r"
            ),
            use_container_width=True,
            height=400,
        )

    with tab3:
        st.markdown("**Download flagged transactions as CSV:**")
        export_df = df_scored[df_scored["fraud_flag"] == 1].sort_values("risk_score", ascending=False)
        csv_data  = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {len(export_df):,} flagged transactions",
            data=csv_data,
            file_name=f"flagged_transactions_{datetime.date.today()}.csv",
            mime="text/csv",
        )
        st.caption(f"Export includes all {len(export_df):,} transactions flagged by the model with risk scores.")

    # ---- Per-Transaction SHAP Explanation ----
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Transaction Deep Dive & Explainability</div>', unsafe_allow_html=True)

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        id_col = None
        for candidate in ["transaction_id", "id", "txn_id"]:
            if candidate in df_scored.columns:
                id_col = candidate
                break

        if id_col:
            options    = df_scored[id_col].tolist()
            selected_id = st.selectbox(f"Select transaction ({id_col})", options, index=0)
            row_idx    = df_scored.index[df_scored[id_col] == selected_id][0]
        else:
            row_idx = st.number_input(
                "Row index (0-based)", min_value=0,
                max_value=len(df_scored) - 1, value=0, step=1,
            )

        row        = df_scored.iloc[row_idx]
        risk_level = row.get("risk_level", "LOW")
        risk_score = row.get("risk_score", 0)
        fraud_prob = row.get("fraud_probability", 0)
        color      = RISK_COLORS.get(risk_level, "#888")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}11);
                    border:1px solid {color}55;border-radius:12px;padding:20px;margin-top:12px;">
            <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;">Risk Assessment</div>
            <div style="font-size:2.5rem;font-weight:700;color:{color};">{risk_level}</div>
            <div style="font-size:1.5rem;font-weight:600;color:#fff;margin-top:4px;">{risk_score:.1f}<span style="font-size:0.9rem;color:#888;"> / 100</span></div>
            <div style="color:#aaa;font-size:0.85rem;margin-top:8px;">Model probability: <b style="color:#fff;">{fraud_prob:.4f}</b></div>
            <div style="color:#aaa;font-size:0.85rem;">Flag status: <b style="color:{'#FF4444' if row.get('fraud_flag',0) else '#00C853'};">{'🚨 FLAGGED' if row.get('fraud_flag',0) else '✅ CLEAR'}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("**Raw transaction data:**")
        display_row = row.drop(labels=[c for c in ["_anomaly_score"] if c in row.index], errors="ignore")
        st.write(display_row.to_frame().T)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Feature-level explanation (SHAP values):**")
    st.caption("Red bars = features pushing toward fraud · Blue bars = features pushing away from fraud")

    with st.spinner("Computing SHAP explanations..."):
        try:
            fig_shap = explain_single_transaction(model, df_scored, row_idx)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    # ---- Auto-refresh simulation ----
    if auto_refresh:
        time.sleep(5)
        st.rerun()

    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#444;font-size:0.8rem;">'
        "RiskShield AI · Behavioral fraud detection · Powered by Random Forest + SHAP"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
