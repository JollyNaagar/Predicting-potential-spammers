#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Project: Fiverr Malicious Spammer & Malware Detection — Production Dashboard
File:    app.py
Author:  Jolly Nagar (ML Engineer)
Description:
    Interactive Streamlit Trust & Safety web application for inspecting model
    benchmarks, diagnostic plots, real-time single-user What-If risk scoring,
    and batch CSV spammer detection.
===============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import custom classes from fiverr_spammer_pipeline.py
from fiverr_spammer_pipeline import (
    FiverrFeatureEngineer,
    SpammerPredictor,
    ModelMetrics,
)

# Register custom classes in __main__ so joblib unpickles artifacts cleanly
setattr(sys.modules["__main__"], "FiverrFeatureEngineer", FiverrFeatureEngineer)
setattr(sys.modules["__main__"], "ModelMetrics", ModelMetrics)

# =============================================================================
# 1. PAGE CONFIGURATION & PATHS
# =============================================================================

st.set_page_config(
    page_title="Fiverr Spammer & Malware Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "fiverr_data.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fiverr_spammer_champion_v1.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_comparison_metrics.csv"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


# =============================================================================
# 2. CACHED RESOURCE LOADERS
# =============================================================================

@st.cache_resource(show_spinner="Loading Champion Model Artifact...")
def load_model_bundle(model_path: Path) -> Dict[str, Any]:
    return joblib.load(model_path)


@st.cache_data(show_spinner="Loading Model Benchmark Metrics...")
def load_metrics_table(metrics_path: Path) -> pd.DataFrame:
    return pd.read_csv(metrics_path)


@st.cache_data(show_spinner="Loading Sample User Telemetry from fiverr_data.csv...")
def load_sample_data(data_path: Path, sample_size: int = 5000) -> pd.DataFrame:
    df = pd.read_csv(data_path, nrows=sample_size)
    return df


# =============================================================================
# 3. SIDEBAR CONTROLS & SYSTEM STATUS
# =============================================================================

st.sidebar.title("🛡️ Trust & Safety Ops")
st.sidebar.markdown("**Model ID:** `MDL-TS-FIVERR-SPAM-v1.0`")
st.sidebar.markdown("**Owner:** Jolly Nagar (ML Engineer)")
st.sidebar.divider()

if not MODEL_PATH.exists():
    st.error(
        f"⚠️ Model artifact not found at `{MODEL_PATH}`. "
        "Please run `fiverr_spammer_pipeline.py` first to generate `artifacts/`."
    )
    st.code(
        r".venv\Scripts\python fiverr_spammer_pipeline.py --data_path fiverr_data.csv --output_dir artifacts",
        language="powershell",
    )
    st.stop()

bundle = load_model_bundle(MODEL_PATH)
pipeline = bundle["pipeline"]
default_opt_thresh = float(bundle.get("optimal_threshold", 0.36))
default_rev_thresh = float(bundle.get("review_threshold", 0.15))
champ_meta = bundle.get("metadata", {})

st.sidebar.subheader("⚙️ Risk Routing Thresholds")
review_threshold = st.sidebar.slider(
    "Tier 2 Step-Up Verification Cutoff",
    min_value=0.05,
    max_value=0.50,
    value=round(default_rev_thresh, 2),
    step=0.01,
    help="Users scoring above this threshold require 2FA challenge and attachment sandboxing.",
)
block_threshold = st.sidebar.slider(
    "Tier 3 Auto-Block Cutoff (Optimal F1)",
    min_value=0.10,
    max_value=0.95,
    value=round(default_opt_thresh, 2),
    step=0.01,
    help="Users scoring above this threshold are automatically blocked from sending attachments.",
)

st.sidebar.divider()
st.sidebar.info(
    "**Three-Tier Routing Policy:**\n"
    f"* 🟢 **Low Risk:** `< {review_threshold:.2f}` (Auto-Allow)\n"
    f"* 🟠 **Medium Risk:** `{review_threshold:.2f} – {block_threshold:.2f}` (Sandbox Briefs)\n"
    f"* 🔴 **High Risk:** `>= {block_threshold:.2f}` (Auto-Block)"
)


# =============================================================================
# 4. MAIN HEADER & NAVIGATION TABS
# =============================================================================

st.title("🛡️ Fiverr Malicious Spammer & Malware Distributor Detection")
st.markdown(
    "Proactive machine learning defense against **Agent Tesla keyloggers** and **Remote Access Trojans (RATs)** "
    "disguised as freelance job brief attachments."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Executive Summary & Benchmarks",
    "📈 2. EDA & Diagnostic Plots",
    "🔍 3. Real-Time User Risk Simulator",
    "📂 4. Batch CSV Threat Scanner",
])


# =============================================================================
# TAB 1: EXECUTIVE SUMMARY & BENCHMARKS
# =============================================================================

with tab1:
    st.subheader("Champion Model Performance on Locked Test Holdout (68,820 Users)")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Champion Architecture", champ_meta.get("model_name", "HistGradientBoosting"))
    k2.metric("Test ROC-AUC", f"{champ_meta.get('roc_auc', 0.0):.4f}")
    k3.metric("Test PR-AUC", f"{champ_meta.get('pr_auc', 0.0):.4f}")
    k4.metric("Precision (@ Opt)", f"{champ_meta.get('precision_at_opt', 0.0):.4f}")
    k5.metric("Recall (@ Opt)", f"{champ_meta.get('recall_at_opt', 0.0):.4f}")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Candidate Model Comparison Table")
        if METRICS_PATH.exists():
            metrics_df = load_metrics_table(METRICS_PATH)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.warning("`model_comparison_metrics.csv` not found in `artifacts/`.")

    with col_right:
        st.markdown("#### Test Set Confusion Matrix (at Optimal Threshold)")
        tn = int(champ_meta.get("tn", 0))
        fp = int(champ_meta.get("fp", 0))
        fn = int(champ_meta.get("fn", 0))
        tp = int(champ_meta.get("tp", 0))

        cm_df = pd.DataFrame(
            [[f"{tn:,} (True Negative)", f"{fp:,} (False Positive)"],
             [f"{fn:,} (False Negative)", f"{tp:,} (True Positive)"]],
            index=["Actual: Legitimate (0)", "Actual: Spammer (1)"],
            columns=["Predicted: Legitimate (0)", "Predicted: Spammer (1)"],
        )
        st.table(cm_df)


# =============================================================================
# TAB 2: EDA & DIAGNOSTIC PLOTS
# =============================================================================

with tab2:
    st.subheader("Exploratory Data Analysis & Model Explainability Artifacts")

    plot_files = [
        ("01_target_distribution.png", "1. Target Class Imbalance (36.24 : 1)"),
        ("02_top_feature_correlations.png", "2. Top 20 Linear Correlations with Spammer Label"),
        ("03_roc_and_pr_curves.png", "3. Multi-Model ROC & Precision-Recall Curves"),
        ("04_threshold_calibration_curve.png", "4. Probability Threshold Calibration Trade-Off"),
        ("05_permutation_feature_importance.png", "5. Top 15 Features by Permutation Importance"),
    ]

    r1_c1, r1_c2 = st.columns(2)
    if (PLOTS_DIR / plot_files[0][0]).exists():
        r1_c1.image(str(PLOTS_DIR / plot_files[0][0]), caption=plot_files[0][1], use_container_width=True)
    if (PLOTS_DIR / plot_files[1][0]).exists():
        r1_c2.image(str(PLOTS_DIR / plot_files[1][0]), caption=plot_files[1][1], use_container_width=True)

    st.divider()
    if (PLOTS_DIR / plot_files[2][0]).exists():
        st.image(str(PLOTS_DIR / plot_files[2][0]), caption=plot_files[2][1], use_container_width=True)

    st.divider()
    r3_c1, r3_c2 = st.columns(2)
    if (PLOTS_DIR / plot_files[3][0]).exists():
        r3_c1.image(str(PLOTS_DIR / plot_files[3][0]), caption=plot_files[3][1], use_container_width=True)
    if (PLOTS_DIR / plot_files[4][0]).exists():
        r3_c2.image(str(PLOTS_DIR / plot_files[4][0]), caption=plot_files[4][1], use_container_width=True)


# =============================================================================
# TAB 3: REAL-TIME USER RISK SIMULATOR (WHAT-IF ANALYSIS)
# =============================================================================

with tab3:
    st.subheader("Interactive User Telemetry Inspector & What-If Simulator")

    if DATA_PATH.exists():
        sample_df = load_sample_data(DATA_PATH, sample_size=5000)

        filter_col, select_col = st.columns([1, 2])
        with filter_col:
            subset_choice = st.radio(
                "Filter Sample Pool By Ground-Truth:",
                options=["All Users", "Confirmed Spammers (label=1)", "Legitimate Users (label=0)"],
            )

        if subset_choice == "Confirmed Spammers (label=1)":
            filtered_pool = sample_df[sample_df["label"] == 1]
        elif subset_choice == "Legitimate Users (label=0)":
            filtered_pool = sample_df[sample_df["label"] == 0]
        else:
            filtered_pool = sample_df

        with select_col:
            selected_user_id = st.selectbox(
                "Select User ID to Inspect:",
                options=filtered_pool["user_id"].tolist(),
            )

        base_row = filtered_pool[filtered_pool["user_id"] == selected_user_id].iloc[0].copy()
        actual_label = int(base_row["label"])
        actual_badge = "🔴 Spammer (1)" if actual_label == 1 else "🟢 Legitimate (0)"

        st.markdown(f"**Selected User ID:** `{int(selected_user_id)}` | **Actual Ground-Truth Label:** {actual_badge}")

        with st.expander("🛠️ Override Top Behavioral Features (What-If Simulation)", expanded=True):
            sc1, sc2, sc3, sc4 = st.columns(4)
            base_row["X19"] = sc1.number_input(
                "X19 (Top Positive Risk Signal)",
                min_value=0,
                max_value=300,
                value=int(base_row["X19"]),
            )
            base_row["X11"] = sc2.selectbox(
                "X11 (Binary Risk Flag)",
                options=[0, 1],
                index=int(base_row["X11"]) if int(base_row["X11"]) in [0, 1] else 0,
            )
            base_row["X4"] = sc3.selectbox(
                "X4 (Referrer / Trust Category)",
                options=[1, 2, 3],
                index=max(0, min(2, int(base_row["X4"]) - 1)),
            )
            base_row["X22"] = sc4.number_input(
                "X22 (Legitimacy Depth Score)",
                min_value=0,
                max_value=100,
                value=int(base_row["X22"]),
            )

        input_df = pd.DataFrame([base_row.drop(labels=["label"])])
        prob_score = float(pipeline.predict_proba(input_df)[:, 1][0])

        if prob_score >= block_threshold:
            action_tier = "🔴 TIER 3: AUTO-BLOCK & QUARANTINE ACCOUNT"
            alert_fn = st.error
        elif prob_score >= review_threshold:
            action_tier = "🟠 TIER 2: STEP-UP 2FA & SANDBOX JOB BRIEF ATTACHMENTS"
            alert_fn = st.warning
        else:
            action_tier = "🟢 TIER 1: AUTO-ALLOW (NORMAL PLATFORM ACCESS)"
            alert_fn = st.success

        res_c1, res_c2 = st.columns([1, 2])
        res_c1.metric("Predicted Spammer Probability", f"{prob_score * 100:.2f}%")
        with res_c2:
            alert_fn(f"**Recommended Trust & Safety Action:** {action_tier}")
    else:
        st.warning(f"Dataset `{DATA_PATH}` not found for interactive sampling.")


# =============================================================================
# TAB 4: BATCH CSV THREAT SCANNER
# =============================================================================

with tab4:
    st.subheader("Batch User Telemetry Scoring & Threat Triage")
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing user telemetry columns (X1–X51) to score:",
        type=["csv"],
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.info(f"Loaded `{len(batch_df):,}` records from uploaded CSV.")
    elif DATA_PATH.exists():
        st.caption("No file uploaded — showing live batch scoring on the first 500 records of `fiverr_data.csv`.")
        batch_df = load_sample_data(DATA_PATH, sample_size=500)
    else:
        batch_df = None

    if batch_df is not None:
        probs = pipeline.predict_proba(batch_df)[:, 1]
        preds = (probs >= block_threshold).astype(int)
        tiers = np.where(
            probs >= block_threshold,
            "TIER_3_AUTO_BLOCK",
            np.where(
                probs >= review_threshold,
                "TIER_2_STEP_UP_AND_SANDBOX",
                "TIER_1_AUTO_ALLOW",
            ),
        )

        scored_df = pd.DataFrame({
            "user_id": batch_df["user_id"] if "user_id" in batch_df.columns else np.arange(len(batch_df)),
            "spammer_probability": np.round(probs, 4),
            "predicted_label": preds,
            "trust_safety_action": tiers,
        })
        if "label" in batch_df.columns:
            scored_df["actual_label"] = batch_df["label"].values

        b1, b2, b3 = st.columns(3)
        b1.metric("🟢 Tier 1: Auto-Allow", f"{int((tiers == 'TIER_1_AUTO_ALLOW').sum()):,}")
        b2.metric("🟠 Tier 2: Step-Up / Sandbox", f"{int((tiers == 'TIER_2_STEP_UP_AND_SANDBOX').sum()):,}")
        b3.metric("🔴 Tier 3: Auto-Block", f"{int((tiers == 'TIER_3_AUTO_BLOCK').sum()):,}")

        st.dataframe(
            scored_df.sort_values(by="spammer_probability", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Scored Threat Report (CSV)",
            data=csv_bytes,
            file_name="fiverr_spammer_batch_predictions.csv",
            mime="text/csv",
        )