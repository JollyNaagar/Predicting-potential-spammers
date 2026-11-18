#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Project: Predict Potential Spammers on Fiverr (Production ML Pipeline)
File:    fiverr_spammer_pipeline.py
Author:  Jolly Nagar (ML Engineer)
Python:  3.10+
Description:
    End-to-end, production-ready Machine Learning pipeline for detecting
    malicious spammers and malware distributors (Agent Tesla / RATs) on Fiverr
    using anonymized behavioral telemetry (fiverr_data.csv).

Usage in VS Code Terminal:
    python fiverr_spammer_pipeline.py --data_path fiverr_data.csv --output_dir artifacts
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configure production logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FiverrSpammerPipeline")


# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

ZERO_VARIANCE_COLS: List[str] = [
    "X27", "X29", "X30", "X33", "X46", "X47", "X48"
]

SPARSE_ACTIVITY_COLS: List[str] = [
    "X13", "X14", "X15", "X19", "X23", "X24", "X25", "X26",
    "X28", "X31", "X32", "X34", "X35", "X36", "X37", "X38",
    "X39", "X40", "X41", "X42", "X43", "X44"
]

ACTIVE_BINARY_COLS: List[str] = [
    "X11", "X12", "X20", "X45", "X49", "X50", "X51"
]

SKEWED_CONTINUOUS_COLS: List[str] = [
    "X1", "X10", "X19", "X21", "X25", "X32"
]


@dataclass
class ModelMetrics:
    """Data container for standardized model evaluation metrics."""
    model_name: str
    train_time_sec: float
    roc_auc: float
    pr_auc: float
    default_threshold: float
    precision_at_05: float
    recall_at_05: float
    f1_at_05: float
    optimal_threshold: float
    precision_at_opt: float
    recall_at_opt: float
    f1_at_opt: float
    tn: int
    fp: int
    fn: int
    tp: int


# =============================================================================
# 2. CUSTOM SCIKIT-LEARN FEATURE ENGINEERING TRANSFORMER
# =============================================================================

class FiverrFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production feature engineering transformer for Fiverr behavioral telemetry.

    Operations performed:
    1. Drops non-predictive identifier (`user_id`) and 7 zero-variance columns
       (`X27`, `X29`, `X30`, `X33`, `X46`, `X47`, `X48`).
    2. Imputes rare missing values in `X13` using the training median.
    3. Constructs domain-specific interaction ratios, differences, collinear
       consolidations, burst activity aggregates, and log1p transformations.
    """

    def __init__(self, drop_zero_variance: bool = True) -> None:
        self.drop_zero_variance = drop_zero_variance
        self.medians_: Dict[str, float] = {}
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "FiverrFeatureEngineer":
        df = X.copy()
        for col in df.columns:
            if col not in ["user_id", "label"] and pd.api.types.is_numeric_dtype(df[col]):
                self.medians_[col] = float(df[col].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Drop identifier and target if accidentally passed
        cols_to_drop = ["user_id", "label"]
        if self.drop_zero_variance:
            cols_to_drop.extend(ZERO_VARIANCE_COLS)
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        # 2. Impute missing values using learned training medians
        for col, med_val in self.medians_.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(med_val)

        # 3. Structural differences and ratios
        if {"X16", "X18"}.issubset(df.columns):
            df["X16_X18_diff"] = df["X16"] - df["X18"]
            df["X16_X18_ratio"] = (df["X16"] + 1.0) / (df["X18"] + 1.0)

        if {"X21", "X22"}.issubset(df.columns):
            df["X21_X22_diff"] = df["X21"] - df["X22"]
            df["X21_X22_ratio"] = (df["X21"] + 1.0) / (df["X22"] + 1.0)

        # 4. Collinear pair consolidations (|r| > 0.93)
        if {"X34", "X35"}.issubset(df.columns):
            df["X34_X35_sum"] = df["X34"] + df["X35"]

        if {"X41", "X44"}.issubset(df.columns):
            df["X41_X44_sum"] = df["X41"] + df["X44"]

        # 5. Sparse activity volume and breadth aggregations
        present_sparse = [c for c in SPARSE_ACTIVITY_COLS if c in df.columns]
        if present_sparse:
            df["total_activity_counts"] = df[present_sparse].sum(axis=1)
            df["nonzero_activity_dims"] = (df[present_sparse] > 0).sum(axis=1)

        # 6. Active binary flags sum
        present_binary = [c for c in ACTIVE_BINARY_COLS if c in df.columns]
        if present_binary:
            df["binary_flags_sum"] = df[present_binary].sum(axis=1)

        # 7. Log1p transformations for heavy right-skewed features
        for col in SKEWED_CONTINUOUS_COLS:
            if col in df.columns:
                df[f"{col}_log1p"] = np.log1p(np.maximum(0.0, df[col].astype(np.float32)))

        # Cast to float32 for memory efficiency on 458k+ rows
        df = df.astype(np.float32)
        self.feature_names_out_ = df.columns.tolist()
        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.feature_names_out_


# =============================================================================
# 3. DATA LOADING & AUTOMATED QUALITY AUDIT
# =============================================================================

def load_and_audit_data(data_path: Path) -> pd.DataFrame:
    """Loads fiverr_data.csv and logs a comprehensive data quality audit."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    logger.info("Loading dataset from %s ...", data_path)
    df = pd.read_csv(data_path)

    total_rows, total_cols = df.shape
    null_count = int(df.isnull().sum().sum())
    null_cols = df.isnull().sum()
    null_cols_dict = null_cols[null_cols > 0].to_dict()

    pos_count = int((df["label"] == 1).sum())
    neg_count = int((df["label"] == 0).sum())
    pos_rate = (pos_count / total_rows) * 100.0
    imbalance_ratio = neg_count / max(1, pos_count)

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    logger.info("=============================================================")
    logger.info("DATA QUALITY AUDIT SUMMARY")
    logger.info("=============================================================")
    logger.info("Total Rows: %d | Total Columns: %d", total_rows, total_cols)
    logger.info("Legitimate Users (label=0): %d (%.4f%%)", neg_count, 100.0 - pos_rate)
    logger.info("Malicious Spammers (label=1): %d (%.4f%%)", pos_count, pos_rate)
    logger.info("Class Imbalance Ratio: %.2f : 1", imbalance_ratio)
    logger.info("Total Missing Values: %d | Columns with Nulls: %s", null_count, null_cols_dict)
    logger.info("Zero-Variance (Constant) Columns (%d): %s", len(constant_cols), constant_cols)
    logger.info("=============================================================")

    return df


# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# =============================================================================

def generate_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates and saves publication-grade EDA charts."""
    eda_dir = output_dir / "plots"
    eda_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(
        ["Legitimate (0)", "Spammer (1)"],
        counts.values,
        color=["#2b5c8f", "#d95f02"],
        edgecolor="black",
    )
    ax.set_title("Fiverr User Class Distribution (36.24:1 Imbalance)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Users")
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100.0
        ax.annotate(
            f"{height:,}\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    plt.savefig(eda_dir / "01_target_distribution.png", dpi=200)
    plt.close()

    # 2. Top Feature Correlations with Target (label)
    numeric_cols = [
        c for c in df.columns
        if c not in ["user_id"] + ZERO_VARIANCE_COLS
    ]
    corrs = df[numeric_cols].corr()["label"].drop("label").dropna().sort_values()
    top_corrs = pd.concat([corrs.head(10), corrs.tail(10)])

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2b5c8f" if v < 0 else "#d95f02" for v in top_corrs.values]
    top_corrs.plot(kind="barh", color=colors, edgecolor="black", ax=ax)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Top 20 Feature Linear Correlations with Spammer Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient (r)")
    plt.tight_layout()
    plt.savefig(eda_dir / "02_top_feature_correlations.png", dpi=200)
    plt.close()

    logger.info("Saved EDA plots to %s", eda_dir)


# =============================================================================
# 5. MODEL TRAINING, THRESHOLD CALIBRATION & EVALUATION
# =============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    beta: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Finds the probability cutoff on the validation Precision-Recall curve
    that maximizes the F_beta score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    beta_sq = beta ** 2
    f_scores = ((1.0 + beta_sq) * precisions * recalls) / (
        (beta_sq * precisions) + recalls + 1e-10
    )
    best_idx = int(np.argmax(f_scores))
    best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])
    return (
        best_thresh,
        float(precisions[best_idx]),
        float(recalls[best_idx]),
        float(f_scores[best_idx]),
    )


def evaluate_candidate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[ModelMetrics, np.ndarray]:
    """Fits a candidate pipeline, calibrates its threshold on validation, and evaluates on test."""
    logger.info("Training model: %s ...", name)
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    # Calibrate optimal threshold on validation set
    opt_thresh, _, _, _ = find_optimal_threshold(y_val, val_probs, beta=1.0)

    # Evaluate on held-out test set at default (0.50) and calibrated threshold
    preds_05 = (test_probs >= 0.50).astype(int)
    preds_opt = (test_probs >= opt_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds_opt).ravel()

    metrics = ModelMetrics(
        model_name=name,
        train_time_sec=round(train_time, 2),
        roc_auc=round(float(roc_auc_score(y_test, test_probs)), 4),
        pr_auc=round(float(average_precision_score(y_test, test_probs)), 4),
        default_threshold=0.50,
        precision_at_05=round(float(precision_score(y_test, preds_05, zero_division=0)), 4),
        recall_at_05=round(float(recall_score(y_test, preds_05, zero_division=0)), 4),
        f1_at_05=round(float(f1_score(y_test, preds_05, zero_division=0)), 4),
        optimal_threshold=round(opt_thresh, 4),
        precision_at_opt=round(float(precision_score(y_test, preds_opt, zero_division=0)), 4),
        recall_at_opt=round(float(recall_score(y_test, preds_opt, zero_division=0)), 4),
        f1_at_opt=round(float(f1_score(y_test, preds_opt, zero_division=0)), 4),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )

    logger.info(
        "%s | ROC-AUC: %.4f | PR-AUC: %.4f | Opt Thresh: %.4f | "
        "Test Precision: %.4f | Test Recall: %.4f | Test F1: %.4f",
        name,
        metrics.roc_auc,
        metrics.pr_auc,
        metrics.optimal_threshold,
        metrics.precision_at_opt,
        metrics.recall_at_opt,
        metrics.f1_at_opt,
    )
    return metrics, test_probs


def generate_evaluation_plots(
    y_test: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    champion_Pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    opt_threshold: float,
    output_dir: Path,
) -> None:
    """Generates ROC, Precision-Recall, Threshold Calibration, and Feature Importance plots."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined ROC & Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.4f})")

        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, linewidth=2, label=f"{name} (AP = {pr_auc:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right", fontsize=9)

    baseline_pos = y_test.mean()
    axes[1].axhline(baseline_pos, color="k", linestyle="--", linewidth=1, label=f"Baseline ({baseline_pos:.3f})")
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_dir / "03_roc_and_pr_curves.png", dpi=200)
    plt.close()

    # 2. Champion Threshold vs Precision / Recall / F1 Trade-off
    champ_val_probs = champion_Pipeline.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, champ_val_probs)
    f1s = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(thresholds, precisions[:-1], label="Precision", color="#2b5c8f", linewidth=2)
    ax.plot(thresholds, recalls[:-1], label="Recall", color="#2ca02c", linewidth=2)
    ax.plot(thresholds, f1s, label="F1-Score", color="#d95f02", linewidth=2.5)
    ax.axvline(opt_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Optimal Threshold ({opt_threshold:.3f})")
    ax.set_title("Champion Model: Probability Threshold vs. Precision, Recall & F1", fontsize=12, fontweight="bold")
    ax.set_xlabel("Classification Probability Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "04_threshold_calibration_curve.png", dpi=200)
    plt.close()

    # 3. Permutation Feature Importance on Validation Subsample
    logger.info("Computing Permutation Feature Importance on validation subsample ...")
    fe_step: FiverrFeatureEngineer = champion_Pipeline.named_steps["feature_engineer"]
    clf_step = champion_Pipeline.named_steps["classifier"]

    X_val_transformed = fe_step.transform(X_val.iloc[:10000])
    y_val_sub = y_val[:10000]

    perm_result = permutation_importance(
        clf_step,
        X_val_transformed,
        y_val_sub,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=2,
    )
    sorted_idx = perm_result.importances_mean.argsort()[-15:]
    top_features = [X_val_transformed.columns[i] for i in sorted_idx]
    top_importances = perm_result.importances_mean[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features, top_importances, color="#2b5c8f", edgecolor="black")
    ax.set_title("Top 15 Features by Permutation Importance (PR-AUC Impact)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Average Precision (PR-AUC)")
    plt.tight_layout()
    plt.savefig(plot_dir / "05_permutation_feature_importance.png", dpi=200)
    plt.close()

    logger.info("Saved evaluation and explainability plots to %s", plot_dir)


# =============================================================================
# 6. PRODUCTION INFERENCE WRAPPER (THREE-TIER RISK ROUTING)
# =============================================================================

class SpammerPredictor:
    """
    Production inference service wrapper that loads the serialized pipeline and
    applies three-tier Trust & Safety risk routing.
    """

    def __init__(self, artifact_path: Path) -> None:
        bundle = joblib.load(artifact_path)
        self.pipeline: Pipeline = bundle["pipeline"]
        self.optimal_threshold: float = bundle["optimal_threshold"]
        self.review_threshold: float = bundle.get("review_threshold", 0.15)
        self.metadata: Dict[str, Any] = bundle.get("metadata", {})

    def predict_risk(self, user_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores incoming user records and assigns a probability, binary prediction,
        and Trust & Safety action tier.
        """
        probs = self.pipeline.predict_proba(user_df)[:, 1]
        preds = (probs >= self.optimal_threshold).astype(int)

        tiers = np.where(
            probs >= self.optimal_threshold,
            "TIER_3_AUTO_BLOCK",
            np.where(
                probs >= self.review_threshold,
                "TIER_2_STEP_UP_VERIFICATION_AND_SANDBOX",
                "TIER_1_AUTO_ALLOW",
            ),
        )

        result = pd.DataFrame({
            "user_id": user_df["user_id"].values if "user_id" in user_df.columns else np.arange(len(user_df)),
            "spammer_probability": np.round(probs, 5),
            "predicted_label": preds,
            "trust_and_safety_action": tiers,
        })
        return result


# =============================================================================
# 7. MAIN EXECUTION WORKFLOW
# =============================================================================

def main(data_path: str, output_dir: str) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Audit Dataset
    df = load_and_audit_data(Path(data_path))

    # Step 2: Generate EDA Visualizations
    generate_eda_plots(df, out_path)

    # Step 3: Stratified Train (70%) / Validation (15%) / Test (15%) Split
    X = df.drop(columns=["label"])
    y = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    logger.info(
        "Stratified Split Sizes -> Train: %d | Validation: %d | Test: %d",
        len(X_train), len(X_val), len(X_test)
    )

    # Step 4: Define Candidate Pipelines
    candidates: Dict[str, Pipeline] = {
        "LogisticRegression_Balanced": Pipeline([
            ("feature_engineer", FiverrFeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=200, C=1.0, random_state=42
            )),
        ]),
        "RandomForest_BalancedSubsample": Pipeline([
            ("feature_engineer", FiverrFeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=5,
                max_samples=0.5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=2,
            )),
        ]),
        "HistGradientBoosting_Champion": Pipeline([
            ("feature_engineer", FiverrFeatureEngineer()),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_leaf_nodes=63,
                min_samples_leaf=30,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=15,
                random_state=42,
            )),
        ]),
    }

    # Step 5: Train & Evaluate Candidates
    all_metrics: List[ModelMetrics] = []
    model_test_probs: Dict[str, np.ndarray] = {}

    for name, pipe in candidates.items():
        metrics, test_probs = evaluate_candidate(
            name, pipe, X_train, y_train, X_val, y_val, X_test, y_test
        )
        all_metrics.append(metrics)
        model_test_probs[name] = test_probs

    # Step 6: Select Champion Model by PR-AUC
    champion_metrics = max(all_metrics, key=lambda m: m.pr_auc)
    champion_name = champion_metrics.model_name
    champion_pipeline = candidates[champion_name]

    logger.info("=============================================================")
    logger.info("CHAMPION MODEL SELECTED: %s", champion_name)
    logger.info("Test ROC-AUC: %.4f | Test PR-AUC: %.4f", champion_metrics.roc_auc, champion_metrics.pr_auc)
    logger.info(
        "Optimal Threshold (%.4f) -> Precision: %.4f | Recall: %.4f | F1: %.4f",
        champion_metrics.optimal_threshold,
        champion_metrics.precision_at_opt,
        champion_metrics.recall_at_opt,
        champion_metrics.f1_at_opt,
    )
    logger.info(
        "Test Confusion Matrix -> TN: %d | FP: %d | FN: %d | TP: %d",
        champion_metrics.tn, champion_metrics.fp, champion_metrics.fn, champion_metrics.tp
    )
    logger.info("=============================================================")

    # Step 7: Save Evaluation Comparison Table & Plots
    metrics_df = pd.DataFrame([asdict(m) for m in all_metrics])
    metrics_csv_path = out_path / "model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Saved model comparison metrics table to %s", metrics_csv_path)

    generate_evaluation_plots(
        y_test=y_test,
        model_probs=model_test_probs,
        champion_Pipeline=champion_pipeline,
        X_val=X_val,
        y_val=y_val,
        opt_threshold=champion_metrics.optimal_threshold,
        output_dir=out_path,
    )

    # Step 8: Serialize Production Model Bundle
    artifact_bundle = {
        "pipeline": champion_pipeline,
        "optimal_threshold": champion_metrics.optimal_threshold,
        "review_threshold": 0.15,
        "metadata": asdict(champion_metrics),
    }
    model_artifact_path = out_path / "fiverr_spammer_champion_v1.joblib"
    joblib.dump(artifact_bundle, model_artifact_path)
    logger.info("Serialized production model bundle to %s", model_artifact_path)

    # Step 9: Smoke-Test Production Inference Wrapper
    predictor = SpammerPredictor(model_artifact_path)
    sample_predictions = predictor.predict_risk(X_test.head(10))
    logger.info("Sample Production Risk Routing Output:\n%s", sample_predictions.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fiverr Spammer Detection ML Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default="fiverr_data.csv",
        help="Path to the input fiverr_data.csv file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory to store trained model artifacts, metrics, and plots",
    )
    args = parser.parse_args()
    main(data_path=args.data_path, output_dir=args.output_dir)