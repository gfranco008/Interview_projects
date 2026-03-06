#!/usr/bin/env python3
"""Train and evaluate a false-positive predictor for Task 2.

Expected inputs
---------------
- Dataset CSV from `generate_dataset.py` (default: `assessment/outputs/alerts_dataset.csv`).
- Policy file with thresholds (default: `assessment/policy.yaml`).

Outputs
-------
- `assessment/outputs/calibration_curve.png`
- `assessment/outputs/roc_curves.png`
- `assessment/outputs/pr_curve.png`
- `assessment/outputs/threshold_analysis.png`
- `assessment/outputs/feature_importance.png`
- `assessment/outputs/shap_summary.png`
- `assessment/outputs/shap_vs_permutation.png`
- `assessment/evaluation.md`

Examples
--------
python3.13 fp_predictor.py
python3.13 fp_predictor.py --extra-columns
python3.13 fp_predictor.py --exclude-escalated --calibration sigmoid
python3.13 fp_predictor.py --include-ioc
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass
class ModelResult:
    name: str
    pipeline: Any
    roc_auc: float
    brier: float
    log_loss: float
    cv_auc: float = 0.0
    cv_std: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    fp_threshold_auto_close: float
    fp_threshold_flag_review: float


def load_policy(policy_path: Path) -> PolicyConfig:
    with policy_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Policy file must contain a mapping.")
    try:
        auto_close = float(raw["fp_threshold_auto_close"])
        flag_review = float(raw["fp_threshold_flag_review"])
    except KeyError as exc:
        raise ValueError(f"Policy missing required key: {exc}") from exc
    if not (np.isfinite(auto_close) and np.isfinite(flag_review)):
        raise ValueError("Policy thresholds must be finite numbers.")
    if not (0.0 <= flag_review <= 1.0 and 0.0 <= auto_close <= 1.0):
        raise ValueError("Policy thresholds must be in [0, 1].")
    if auto_close < flag_review:
        raise ValueError("fp_threshold_auto_close must be >= fp_threshold_flag_review.")
    return PolicyConfig(
        fp_threshold_auto_close=auto_close,
        fp_threshold_flag_review=flag_review,
    )


def build_features(
    df: pd.DataFrame,
    include_extras: bool,
    include_ioc: bool = False,
) -> Tuple[List[str], List[str]]:
    missing_required = [
        col
        for col in ["contextual_risk", "burst_index", "severity_gap"]
        if col not in df.columns
    ]
    if missing_required:
        raise ValueError(
            "Derived feature(s) missing from dataset: "
            f"{', '.join(missing_required)}. Re-run generate_dataset.py."
        )
    numeric_features = [
        "inherent_severity",
        "vendor_severity",
        "asset_criticality",
        "alert_count_1h",
        "alert_count_24h",
        "time_since_last_alert",
        "contextual_risk",
        "burst_index",
        "severity_gap",
        "low_crit_burst",
    ]
    categorical_features = [
        "source_rule",
        "mitre_tactic",
        "mitre_technique",
        "asset_type",
        "user_privilege_level",
    ]

    if include_ioc:
        col = "has_ioc_match"
        if col not in df.columns:
            raise ValueError(
                f"IOC-derived feature '{col}' missing from dataset. "
                "Re-run generate_dataset.py or omit --include-ioc."
            )
        numeric_features.append(col)

    if include_extras:
        extra_categoricals = [
            "alert_type",
            "detection_tool",
            "detection_vendor",
            "user_department",
            "process_name",
            "network_direction",
        ]
        for col in extra_categoricals:
            if col in df.columns:
                categorical_features.append(col)

    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    return numeric_features, categorical_features


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric, preserving NaNs for imputers."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_boolean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize boolean-like columns to 0/1 integers."""
    df = df.copy()
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y", "t"])
            .astype(int)
        )
    return df


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def fit_and_score(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cv_auc: float = 0.0,
    cv_std: float = 0.0,
) -> ModelResult:
    pipeline.fit(x_train, y_train)
    proba = pipeline.predict_proba(x_val)[:, 1]
    return ModelResult(
        name=name,
        pipeline=pipeline,
        roc_auc=roc_auc_score(y_val, proba),
        brier=brier_score_loss(y_val, proba),
        log_loss=log_loss(y_val, proba),
        cv_auc=cv_auc,
        cv_std=cv_std,
    )


def compute_cv_auc(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, seed: int
) -> Tuple[float, float]:
    """Compute 5-fold cross-validated ROC-AUC for model comparison."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, x, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))


def tune_xgboost(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
) -> Tuple[ModelResult, dict]:
    """Run a small grid search for XGBoost hyperparameters and score on validation."""
    xgb = Pipeline([
        ("preprocess", preprocessor),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ])
    param_grid = {
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.08, 0.12],
        "model__n_estimators": [150, 200, 250],
    }
    grid = GridSearchCV(
        xgb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(x_train, y_train)
    best_pipeline = grid.best_estimator_
    best_index = grid.best_index_
    best_cv_std = float(grid.cv_results_["std_test_score"][best_index])
    prob_val = best_pipeline.predict_proba(x_val)[:, 1]
    result = ModelResult(
        name="XGBoost (tuned)",
        pipeline=best_pipeline,
        roc_auc=roc_auc_score(y_val, prob_val),
        brier=brier_score_loss(y_val, prob_val),
        log_loss=log_loss(y_val, prob_val),
        cv_auc=float(grid.best_score_),
        cv_std=best_cv_std,
    )
    tuning_summary = {
        "param_grid": param_grid,
        "best_params": grid.best_params_,
        "best_cv_auc": float(grid.best_score_),
        "best_cv_std": best_cv_std,
    }
    return result, tuning_summary


def plot_calibration(y_true, prob_pre, prob_post, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")

    frac_pos, mean_pred = calibration_curve(y_true, prob_pre, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", label="Pre-calibration")

    frac_pos_c, mean_pred_c = calibration_curve(y_true, prob_post, n_bins=10)
    ax.plot(mean_pred_c, frac_pos_c, marker="o", label="Post-calibration")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc(
    model_results: List[ModelResult], x_test, y_test, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in model_results:
        proba = result.pipeline.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(
            fpr, tpr, label=f"{result.name} (AUC {roc_auc_score(y_test, proba):.2f})"
        )
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    model_results: List[ModelResult], x_test, y_test, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in model_results:
        proba = result.pipeline.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f"{result.name} (AP {ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_threshold_analysis(probabilities, y_true, output_path: Path) -> None:
    thresholds = np.linspace(0.5, 0.99, 20)
    fp_pass = []
    tp_missed = []
    workload = []

    y_true = np.asarray(y_true)
    fp_total = max(1, np.sum(y_true == 1))
    tp_total = max(1, np.sum(y_true == 0))
    for threshold in thresholds:
        auto_close = probabilities >= threshold
        fp_pass.append(np.sum((y_true == 1) & ~auto_close) / fp_total)
        tp_missed.append(np.sum((y_true == 0) & auto_close) / tp_total)
        workload.append(np.mean(~auto_close))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, fp_pass, label="FP pass-through")
    ax.plot(thresholds, tp_missed, label="Missed TPs")
    ax.plot(thresholds, workload, label="Analyst workload")
    ax.set_xlabel("Auto-close threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Threshold Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_threshold_metrics(probabilities, y_true, policy: PolicyConfig) -> dict:
    auto_close = policy.fp_threshold_auto_close
    flag_review = policy.fp_threshold_flag_review

    probs = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    auto_mask = probs >= auto_close
    flag_mask = (probs >= flag_review) & (probs < auto_close)
    triage_mask = probs < flag_review

    fp_total = max(1, np.sum(y_true == 1))
    tp_total = max(1, np.sum(y_true == 0))
    return {
        "fp_pass_through": np.sum((y_true == 1) & ~auto_mask) / fp_total,
        "missed_tp_rate": np.sum((y_true == 0) & auto_mask) / tp_total,
        "auto_close_rate": float(np.mean(auto_mask)),
        "flag_review_rate": float(np.mean(flag_mask)),
        "manual_triage_rate": float(np.mean(triage_mask)),
        "workload_rate": float(np.mean(flag_mask | triage_mask)),
    }


def write_evaluation_report(
    output_path: Path,
    best_result: ModelResult,
    model_results: List[ModelResult],
    calibrated_result: ModelResult,
    calibration_metrics: dict,
    importance_summary: dict,
    threshold_metrics: dict,
    severity_conflict: dict,
    operating_point: dict,
    error_analysis: dict,
    xgb_tuning: dict,
    shap_summary: dict,
    artifacts: dict,
) -> None:
    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Model Comparison")
    for result in model_results:
        cv_text = (
            f"CV AUC {result.cv_auc:.3f}±{result.cv_std:.3f}"
            if result.cv_auc > 0
            else "CV AUC n/a"
        )
        lines.append(
            f"- {result.name}: {cv_text}, validation AUC {result.roc_auc:.3f}."
        )
    lines.append("Note: ROC/PR plots use uncalibrated model scores for comparison.")
    lines.append("PR curve: outputs/pr_curve.png")
    selected_cv = (
        f"CV AUC {best_result.cv_auc:.3f}±{best_result.cv_std:.3f}"
        if best_result.cv_auc > 0
        else "CV AUC n/a"
    )
    lines.append(
        f"Selected model: **{best_result.name}** "
        f"({selected_cv}, validation AUC {best_result.roc_auc:.3f})."
    )
    lines.append(
        f"Calibrated model test AUC: {calibrated_result.roc_auc:.3f}, "
        f"Brier: {calibrated_result.brier:.3f}, Log loss: {calibrated_result.log_loss:.3f}."
    )
    lines.append(
        "Calibration metrics (test): "
        f"pre Brier {calibration_metrics['pre_brier']:.3f}, "
        f"pre LogLoss {calibration_metrics['pre_log_loss']:.3f}; "
        f"post Brier {calibration_metrics['post_brier']:.3f}, "
        f"post LogLoss {calibration_metrics['post_log_loss']:.3f}."
    )
    lines.append(
        "Calibration trust note: miscalibrated scores erode analyst confidence because "
        "probabilities no longer map to observed FP rates, leading to inconsistent triage."
    )
    lines.append(
        "Feature importance check: inherent_severity "
        f"{importance_summary['inherent_severity']:.4f} vs vendor_severity "
        f"{importance_summary['vendor_severity']:.4f} (permutation importance mean)."
    )
    lines.append(
        "Calibration strategy: CalibratedClassifierCV with FrozenEstimator, fit on the "
        "validation split to preserve the exact base model weights."
    )
    lines.append(
        "Note: validation was reused for both model comparison and calibration in this "
        "take-home setup. In production, I would separate tuning, calibration, and final "
        "evaluation with either a dedicated calibration holdout or time-based backtesting."
    )
    lines.append(
        "Note: test set is 20% of data; subset metrics (e.g., severity conflicts) "
        "can be noisy for rare segments."
    )
    lines.append("")
    lines.append("## Threshold Policy Impact")
    lines.append(
        f"Auto-close threshold: {threshold_metrics['auto_close']:.2f}, "
        f"Flag-review threshold: {threshold_metrics['flag_review']:.2f}"
    )
    lines.append(
        f"False-positive pass-through rate: {threshold_metrics['fp_pass_through']:.2%}"
    )
    lines.append(
        f"Missed true-positive rate: {threshold_metrics['missed_tp_rate']:.2%}"
    )
    lines.append(f"Analyst workload rate: {threshold_metrics['workload_rate']:.2%}")
    lines.append(
        "Volume split: "
        f"auto-close {threshold_metrics['auto_close_rate']:.1%}, "
        f"flag-review {threshold_metrics['flag_review_rate']:.1%}, "
        f"manual triage {threshold_metrics['manual_triage_rate']:.1%}."
    )
    lines.append("")
    lines.append("## Operating Point Metrics")
    lines.append(
        f"Chosen threshold: {operating_point['threshold']:.2f} "
        f"(policy flag-review boundary)"
    )
    lines.append(
        f"Precision: {operating_point['precision']:.3f}, "
        f"Recall: {operating_point['recall']:.3f}, "
        f"F1: {operating_point['f1']:.3f}"
    )
    lines.append(
        "Justification: aligns with the review boundary so predictions above the "
        "threshold drive analyst action while minimizing missed TPs at auto-close."
    )
    lines.append("")
    lines.append("## Hyperparameter Tuning")
    lines.append(
        "Used GridSearchCV (5-fold ROC-AUC) over XGBoost depth, learning_rate, and "
        "number of estimators."
    )
    clean_grid = {
        k.replace("model__", ""): v for k, v in xgb_tuning["param_grid"].items()
    }
    clean_best = {
        k.replace("model__", ""): v for k, v in xgb_tuning["best_params"].items()
    }
    lines.append(
        "Grid: "
        f"max_depth={clean_grid['max_depth']}, "
        f"learning_rate={clean_grid['learning_rate']}, "
        f"n_estimators={clean_grid['n_estimators']}."
    )
    lines.append(
        "Best params: "
        f"{clean_best} (CV AUC {xgb_tuning['best_cv_auc']:.3f}±"
        f"{xgb_tuning['best_cv_std']:.3f})."
    )
    lines.append("")
    lines.append("## Severity Conflict Analysis")
    lines.append("Subset where |inherent_severity - vendor_severity| >= 2:")
    overall = severity_conflict["overall"]
    lines.append(
        f"Count: {overall['count']}, "
        f"FP rate: {overall['fp_rate']:.2%}, "
        f"AUC: {overall['auc']:.3f}"
    )
    lines.append("")
    lines.append("High vendor / low inherent (vendor 4-5, inherent 1):")
    high_v_low_i = severity_conflict["high_vendor_low_inherent"]
    lines.append(
        f"Count: {high_v_low_i['count']}, "
        f"Precision: {high_v_low_i['precision']:.3f}, "
        f"Recall: {high_v_low_i['recall']:.3f}, "
        f"AUC: {high_v_low_i['auc']:.3f}"
    )
    lines.append("Low vendor / high inherent (vendor 1-2, inherent 3-4):")
    low_v_high_i = severity_conflict["low_vendor_high_inherent"]
    lines.append(
        f"Count: {low_v_high_i['count']}, "
        f"Precision: {low_v_high_i['precision']:.3f}, "
        f"Recall: {low_v_high_i['recall']:.3f}, "
        f"AUC: {low_v_high_i['auc']:.3f}"
    )
    lines.append("")
    lines.append("Qualitative interpretation:")
    lines.append(severity_conflict["interpretation"])
    lines.append(
        "Vendor-severity anchoring bias failure mode: if the model overweights vendor "
        "scores, high vendor/low inherent alerts would be misclassified as true positives; "
        "we explicitly check the conflicting subsets above to guard against that bias."
    )
    lines.append("")
    lines.append("## Error Analysis")
    overall_errors = error_analysis["overall"]
    lines.append(
        f"Overall error rate: {overall_errors['error_rate']:.2%} "
        f"(FN {overall_errors['fn_count']}, FP {overall_errors['fp_count']})."
    )
    for group_col, group_stats in error_analysis["groups"].items():
        lines.append(f"By {group_col}:")
        lines.append("Missed FPs (FN): " + format_group_stats(group_stats["fn"]))
        lines.append(
            "True alerts flagged as FP (FP): " + format_group_stats(group_stats["fp"])
        )
    lines.append(error_analysis["conclusion"])
    lines.append("")
    lines.append("## SHAP Explainability")
    if shap_summary.get("available"):
        plot_path = Path(shap_summary["plot_path"])
        plot_name = (
            plot_path.name if plot_path.exists() else "shap_summary.png (missing)"
        )
        lines.append(
            f"SHAP summary computed on {shap_summary['sample_size']} test samples."
        )
        lines.append(f"Plot: outputs/{plot_name}")
        top_features = ", ".join(
            f"{item['feature']} ({item['mean_abs_shap']:.4f})"
            for item in shap_summary["top_features"]
        )
        lines.append(f"Top features by mean(|SHAP|): {top_features}")
        comparison_plot = shap_summary.get("comparison_plot")
        if comparison_plot and comparison_plot.get("available"):
            comparison_path = Path(comparison_plot["path"])
            comparison_name = (
                comparison_path.name
                if comparison_path.exists()
                else "shap_vs_permutation.png (missing)"
            )
            lines.append(
                "SHAP vs permutation comparison: "
                f"outputs/{comparison_name} (normalized scale)."
            )
    else:
        lines.append(f"SHAP not generated: {shap_summary.get('reason', 'unknown')}.")
    lines.append("")
    lines.append("## Saved Artifacts")
    lines.append(f"Base model: {artifacts['base_model']}")
    lines.append(f"Calibrated model: {artifacts['calibrated_model']}")
    lines.append(f"Metadata: {artifacts['metadata']}")
    lines.append(f"Policy snapshot: {artifacts['policy_snapshot']}")
    lines.append("")
    lines.append("## SOC Deployment Notes")
    lines.append(
        "If I were deploying this to a SOC handling ~500 alerts/day, I would position the "
        "model as a decision-support layer that drives the triage split. Alerts with a "
        f"predicted FP probability >= {threshold_metrics['flag_review']:.2f} move to a "
        "fast-path review queue, and those above the auto-close threshold require the "
        "lowest manual effort. With the current policy, the expected analyst workload "
        f"is about {threshold_metrics['workload_rate']:.1%} of daily volume; that maps "
        "to a concrete staffing estimate and is easy to tune by adjusting the thresholds."
    )
    lines.append(
        "I would tell analysts that the model is calibrated (post-calibration Brier/log-loss "
        "tracked) and that we explicitly monitor error hot spots by tactic, asset type, and "
        "rule. If we see drift (e.g., bursts in a specific tactic), we can temporarily "
        "override auto-close for that group and feed the analyst feedback into the next "
        "retraining cycle. The goal is not to replace judgment, but to prioritize review "
        "where the model is least confident or historically error-prone."
    )
    lines.append(
        "Operationally, I would start with conservative thresholds in week 1, review the "
        "false-negative cases daily, and only then increase auto-close aggressiveness. "
        "We would also validate vendor/inherent severity conflicts and make sure the model "
        "behavior aligns with analyst expectations for high-impact tactics. This keeps the "
        "system transparent, adjustable, and aligned with SOC risk tolerance."
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_auc(y_true, prob) -> float:
    """Compute ROC AUC safely for small or single-class subsets."""
    if len(y_true) == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, prob))


def compute_subset_metrics(
    mask: pd.Series,
    x: pd.DataFrame,
    y: pd.Series,
    model: Any,
    threshold: float,
) -> dict:
    """Compute precision/recall and AUC for a severity-conflict subset."""
    subset_y = y.loc[mask]
    if subset_y.empty:
        return {
            "count": 0,
            "fp_rate": 0.0,
            "auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_prob": 0.0,
        }

    subset_x = x.loc[mask]
    prob = model.predict_proba(subset_x)[:, 1]
    y_pred = (prob >= threshold).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(
        subset_y, y_pred, pos_label=1, average="binary", zero_division=0
    )
    return {
        "count": int(mask.sum()),
        "fp_rate": float(np.mean(subset_y == 1)),
        "auc": safe_auc(subset_y, prob),
        "precision": float(precision),
        "recall": float(recall),
        "mean_prob": float(np.mean(prob)),
    }


def summarize_error_groups(
    errors: pd.DataFrame, group_col: str, error_mask: pd.Series, top_n: int = 5
) -> List[dict]:
    """Summarize misclassification concentration for a group column."""
    if group_col not in errors.columns:
        return []

    totals = errors.groupby(group_col).size()
    err_counts = errors[error_mask].groupby(group_col).size()
    rows = []
    for group, total in totals.items():
        count = int(err_counts.get(group, 0))
        if count == 0:
            continue
        rows.append({
            "group": str(group),
            "count": count,
            "total": int(total),
            "rate": count / max(1, int(total)),
        })
    rows.sort(key=lambda r: (r["count"], r["rate"]), reverse=True)
    return rows[:top_n]


def format_group_stats(stats: List[dict]) -> str:
    """Format grouped error stats into a compact line."""
    if not stats:
        return "None"
    return ", ".join(
        f"{item['group']} {item['count']}/{item['total']} ({item['rate']:.1%})"
        for item in stats
    )


def summarize_error_conclusion(error_analysis: dict) -> str:
    """Create a short conclusion about where errors concentrate."""
    best = None
    for group_col, group_stats in error_analysis.get("groups", {}).items():
        for kind in ("fp", "fn"):
            stats = group_stats.get(kind, [])
            if not stats:
                continue
            top = stats[0]
            if best is None or (top["count"], top["rate"]) > (
                best[4]["count"],
                best[4]["rate"],
            ):
                best = (top["count"], top["rate"], group_col, kind, top)

    if best is None:
        return "Conclusion: no dominant error concentration detected in the grouped analysis."

    _, _, group_col, kind, top = best
    label = "FPs" if kind == "fp" else "FNs"
    if group_col == "source_rule":
        reason = "likely driven by a noisy or overly broad rule definition."
    elif group_col == "mitre_tactic":
        reason = "likely due to broader detection logic within that tactic."
    elif group_col == "asset_type":
        reason = "likely due to environmental noise specific to that asset class."
    else:
        reason = "likely due to heterogeneous behavior in that group."
    return (
        f"Conclusion: {label} concentrate most in {group_col} '{top['group']}' "
        f"({top['count']}/{top['total']}, {top['rate']:.1%}), {reason}"
    )


def compute_shap_summary(
    pipeline: Pipeline,
    x: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> dict:
    """Compute SHAP values for a sample of the test set and save a summary plot."""
    try:
        import shap  # type: ignore
    except ImportError:
        return {"available": False, "reason": "shap not installed"}

    if x.empty:
        return {"available": False, "reason": "empty test set"}

    rng = np.random.default_rng(seed)
    if len(x) > sample_size:
        sample_idx = rng.choice(len(x), size=sample_size, replace=False)
        x_sample = x.iloc[sample_idx]
    else:
        x_sample = x

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    try:
        x_trans = preprocess.transform(x_sample)
        feature_names = preprocess.get_feature_names_out()
        if isinstance(model, XGBClassifier):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            # Using the sample as background data for LinearExplainer is acceptable but
            # yields correlation-based attributions rather than interventional ones.
            explainer = shap.LinearExplainer(model, x_trans)
        else:
            return {"available": False, "reason": "unsupported model for SHAP"}

        shap_values = explainer.shap_values(x_trans)
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]
        else:
            shap_values_plot = shap_values

        plot_path = output_dir / "shap_summary.png"
        shap.summary_plot(
            shap_values_plot,
            x_trans,
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        mean_abs = np.mean(np.abs(shap_values_plot), axis=0)
        top_idx = np.argsort(mean_abs)[-10:][::-1]
        top_features = [
            {"feature": str(feature_names[i]), "mean_abs_shap": float(mean_abs[i])}
            for i in top_idx
        ]

        return {
            "available": True,
            "plot_path": str(plot_path),
            "top_features": top_features,
            "feature_names": [str(name) for name in feature_names],
            "mean_abs_shap": [float(val) for val in mean_abs],
            "sample_size": int(len(x_sample)),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"shap error: {exc.__class__.__name__}",
        }


def plot_shap_vs_permutation(
    shap_summary: dict,
    feature_names: List[str],
    perm_importance: Any,
    output_path: Path,
    top_n: int = 10,
) -> dict:
    """Compare SHAP mean(|value|) vs permutation importance and save a plot."""
    if not shap_summary.get("available"):
        return {
            "available": False,
            "reason": shap_summary.get("reason", "shap unavailable"),
        }

    perm_means = np.asarray(perm_importance.importances_mean)
    if perm_means.size == 0:
        return {"available": False, "reason": "empty permutation importances"}

    n_features = min(len(feature_names), len(perm_means))
    if n_features == 0:
        return {"available": False, "reason": "no features available for comparison"}
    perm_means = perm_means[:n_features]
    perm_feature_names = feature_names[:n_features]

    perm_top_idx = np.argsort(perm_means)[-top_n:][::-1]
    perm_top_names = [perm_feature_names[i] for i in perm_top_idx]
    shap_top_names = [
        item["feature"] for item in shap_summary.get("top_features", [])[:top_n]
    ]
    combined = list(dict.fromkeys(perm_top_names + shap_top_names))
    if not combined:
        return {"available": False, "reason": "no overlapping features to plot"}

    perm_lookup = {
        name: float(value)
        for name, value in zip(perm_feature_names, perm_means)
    }
    if "mean_abs_shap" in shap_summary and "feature_names" in shap_summary:
        shap_lookup = {
            str(name): float(value)
            for name, value in zip(
                shap_summary.get("feature_names", []),
                shap_summary.get("mean_abs_shap", []),
            )
        }
    else:
        shap_lookup = {
            item["feature"]: float(item["mean_abs_shap"])
            for item in shap_summary.get("top_features", [])
        }

    perm_vals = np.array([perm_lookup.get(name, 0.0) for name in combined], dtype=float)
    shap_vals = np.array([shap_lookup.get(name, 0.0) for name in combined], dtype=float)

    perm_scale = np.max(perm_vals) if np.max(perm_vals) > 0 else 1.0
    shap_scale = np.max(shap_vals) if np.max(shap_vals) > 0 else 1.0
    perm_norm = perm_vals / perm_scale
    shap_norm = shap_vals / shap_scale

    y = np.arange(len(combined))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(combined))))
    ax.barh(y - height / 2, perm_norm, height=height, label="Permutation (normalized)")
    ax.barh(
        y + height / 2,
        shap_norm,
        height=height,
        label="SHAP mean(|value|) (normalized)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(combined)
    ax.set_xlabel("Normalized importance")
    ax.set_title("SHAP vs Permutation Importance (Top Features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return {"available": True, "path": str(output_path)}


def save_artifacts(
    output_dir: Path,
    base_model: Pipeline,
    calibrated_model: Any,
    metadata: dict,
    policy: PolicyConfig,
) -> dict:
    """Persist model artifacts and metadata for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "model_base.joblib"
    calibrated_path = output_dir / "model_calibrated.joblib"
    metadata_path = output_dir / "model_metadata.json"
    policy_path = output_dir / "policy_snapshot.yaml"

    joblib.dump(base_model, base_path)
    joblib.dump(calibrated_model, calibrated_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(asdict(policy)), encoding="utf-8")

    return {
        "base_model": str(base_path),
        "calibrated_model": str(calibrated_path),
        "metadata": str(metadata_path),
        "policy_snapshot": str(policy_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FP predictor and evaluate")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "alerts_dataset.csv",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parent / "policy.yaml",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-escalated", action="store_true")
    parser.add_argument("--extra-columns", action="store_true")
    parser.add_argument(
        "--include-ioc",
        action="store_true",
        help=(
            "Include IOC-derived features (has_ioc_match). "
            "Off by default: this feature is a near-perfect label proxy that "
            "causes leakage and degrades on customers with incomplete IOC coverage."
        ),
    )
    parser.add_argument(
        "--calibration", choices=["sigmoid", "isotonic"], default="isotonic"
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset}. Run generate_dataset.py first."
        )

    df = pd.read_csv(args.dataset)
    if args.exclude_escalated:
        df = df[df["disposition"] != "escalated"].copy()

    df = normalize_boolean(df, "has_ioc_match")
    if "low_crit_burst" not in df.columns:
        burst = pd.to_numeric(df["burst_index"], errors="coerce")
        crit = pd.to_numeric(df["asset_criticality"], errors="coerce")
        df["low_crit_burst"] = burst * (1 - crit / 10)

    y = (df["disposition"] == "false_positive").astype(int)
    x = df.drop(columns=["disposition", "alert_id", "timestamp"], errors="ignore")

    numeric_features, categorical_features = build_features(
        df,
        args.extra_columns,
        include_ioc=args.include_ioc,
    )
    df = coerce_numeric(df, numeric_features)
    x = df.drop(columns=["disposition", "alert_id", "timestamp"], errors="ignore")
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=args.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=args.seed
    )

    log_reg = Pipeline([
        ("preprocess", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    xgb_result, xgb_tuning = tune_xgboost(
        preprocessor, x_train, y_train, x_val, y_val, args.seed
    )

    # cross_val_score fits fresh clones of the pipeline; ok to run before log_reg is fitted.
    logreg_cv_auc, logreg_cv_std = compute_cv_auc(log_reg, x_train, y_train, args.seed)
    results = [
        fit_and_score(
            "Logistic Regression",
            log_reg,
            x_train,
            y_train,
            x_val,
            y_val,
            cv_auc=logreg_cv_auc,
            cv_std=logreg_cv_std,
        ),
        xgb_result,
    ]
    results_sorted = sorted(
        results, key=lambda r: r.cv_auc if r.cv_auc > 0 else r.roc_auc, reverse=True
    )
    best = results_sorted[0]

    # FrozenEstimator preserves the exact fitted model; calibration is fit on x_val.
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(best.pipeline),
        method=args.calibration,
    )
    calibrated_model.fit(x_val, y_val)

    prob_pre = best.pipeline.predict_proba(x_test)[:, 1]
    prob_post = calibrated_model.predict_proba(x_test)[:, 1]
    calibration_metrics = {
        "pre_brier": float(brier_score_loss(y_test, prob_pre)),
        "pre_log_loss": float(log_loss(y_test, prob_pre)),
        "post_brier": float(brier_score_loss(y_test, prob_post)),
        "post_log_loss": float(log_loss(y_test, prob_post)),
    }

    calibrated_metrics = ModelResult(
        name=f"{best.name} (calibrated)",
        pipeline=calibrated_model,
        roc_auc=roc_auc_score(y_test, prob_post),
        brier=calibration_metrics["post_brier"],
        log_loss=calibration_metrics["post_log_loss"],
    )

    policy = load_policy(args.policy)
    threshold_metrics = compute_threshold_metrics(prob_post, y_test, policy)
    threshold_metrics["auto_close"] = policy.fp_threshold_auto_close
    threshold_metrics["flag_review"] = policy.fp_threshold_flag_review
    chosen_threshold = policy.fp_threshold_flag_review
    y_pred = (prob_post >= chosen_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, pos_label=1, average="binary", zero_division=0
    )
    operating_point = {
        "threshold": float(chosen_threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    conflict_mask = (df["inherent_severity"] - df["vendor_severity"]).abs() >= 2
    high_v_low_i_mask = (df["vendor_severity"] >= 4) & (df["inherent_severity"] == 1)
    low_v_high_i_mask = (df["vendor_severity"] <= 2) & (df["inherent_severity"] >= 3)

    conflict_mask_test = conflict_mask.loc[x_test.index]
    high_v_low_i_mask_test = high_v_low_i_mask.loc[x_test.index]
    low_v_high_i_mask_test = low_v_high_i_mask.loc[x_test.index]

    conflict_overall = compute_subset_metrics(
        conflict_mask_test, x_test, y_test, calibrated_model, chosen_threshold
    )
    high_v_low_i = compute_subset_metrics(
        high_v_low_i_mask_test, x_test, y_test, calibrated_model, chosen_threshold
    )
    low_v_high_i = compute_subset_metrics(
        low_v_high_i_mask_test, x_test, y_test, calibrated_model, chosen_threshold
    )

    interpretation = "Insufficient samples to determine severity influence."
    if high_v_low_i["count"] > 0 and low_v_high_i["count"] > 0:
        delta = high_v_low_i["mean_prob"] - low_v_high_i["mean_prob"]
        if abs(delta) < 0.02:
            interpretation = (
                "Model appears balanced between vendor and inherent severity "
                f"(mean FP prob delta={delta:+.3f})."
            )
        elif delta > 0:
            interpretation = (
                "Model leans toward inherent_severity: higher FP probabilities when "
                "inherent is low even if vendor severity is high "
                f"(mean FP prob delta={delta:+.3f})."
            )
        else:
            interpretation = (
                "Model leans toward vendor_severity: higher FP probabilities when "
                "vendor severity is low even if inherent severity is high "
                f"(mean FP prob delta={delta:+.3f})."
            )

    severity_conflict = {
        "overall": conflict_overall,
        "high_vendor_low_inherent": high_v_low_i,
        "low_vendor_high_inherent": low_v_high_i,
        "interpretation": interpretation,
    }

    errors = x_test.copy()
    for col in ["mitre_tactic", "asset_type", "source_rule"]:
        if col in df.columns:
            errors[col] = df.loc[x_test.index, col].values
    errors["y_true"] = y_test.values
    errors["y_pred"] = (prob_post >= chosen_threshold).astype(int)
    errors["prob"] = prob_post

    fn_mask = (errors["y_true"] == 1) & (errors["y_pred"] == 0)
    fp_mask = (errors["y_true"] == 0) & (errors["y_pred"] == 1)
    error_analysis = {
        "overall": {
            "error_rate": float(np.mean(errors["y_true"] != errors["y_pred"])),
            "fn_count": int(fn_mask.sum()),
            "fp_count": int(fp_mask.sum()),
        },
        "groups": {},
    }

    for group_col in ["mitre_tactic", "asset_type", "source_rule"]:
        error_analysis["groups"][group_col] = {
            "fn": summarize_error_groups(errors, group_col, fn_mask),
            "fp": summarize_error_groups(errors, group_col, fp_mask),
        }
    error_analysis["conclusion"] = summarize_error_conclusion(error_analysis)

    args.outputs.mkdir(parents=True, exist_ok=True)
    plot_calibration(
        y_test, prob_pre, prob_post, args.outputs / "calibration_curve.png"
    )
    plot_roc(results_sorted, x_test, y_test, args.outputs / "roc_curves.png")
    plot_pr_curve(results_sorted, x_test, y_test, args.outputs / "pr_curve.png")
    plot_threshold_analysis(prob_post, y_test, args.outputs / "threshold_analysis.png")
    shap_summary = compute_shap_summary(
        best.pipeline,
        x_test,
        args.outputs,
        sample_size=1000,
        seed=args.seed,
    )

    metadata = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": str(args.dataset),
        "model_name": best.name,
        "calibration_method": args.calibration,
        "include_ioc": bool(args.include_ioc),
        "include_extras": bool(args.extra_columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "split_sizes": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "cv_auc": float(best.cv_auc),
        "cv_std": float(best.cv_std),
        "xgb_best_params": {
            k.replace("model__", ""): v
            for k, v in (xgb_tuning.get("best_params") or {}).items()
        },
    }
    artifacts = save_artifacts(
        args.outputs, best.pipeline, calibrated_model, metadata, policy
    )

    perm = permutation_importance(
        best.pipeline,
        x_test,
        y_test,
        n_repeats=8,
        random_state=args.seed,
        scoring="roc_auc",
    )

    feature_names = list(
        best.pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    shap_summary["comparison_plot"] = plot_shap_vs_permutation(
        shap_summary,
        feature_names,
        perm,
        args.outputs / "shap_vs_permutation.png",
        top_n=10,
    )

    def mean_importance(feature_suffix: str) -> float:
        idx = [
            i
            for i, name in enumerate(feature_names)
            if name == f"num__{feature_suffix}"
        ]
        if not idx:
            return 0.0
        return float(np.mean(perm.importances_mean[idx]))

    importance_summary = {
        "inherent_severity": mean_importance("inherent_severity"),
        "vendor_severity": mean_importance("vendor_severity"),
    }
    top_idx = np.argsort(perm.importances_mean)[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in top_idx],
        perm.importances_mean[top_idx],
    )
    ax.set_title("Top Feature Importances (Permutation)")
    fig.tight_layout()
    fig.savefig(args.outputs / "feature_importance.png", dpi=150)
    plt.close(fig)

    report_path = Path(__file__).resolve().parent / "evaluation.md"
    write_evaluation_report(
        report_path,
        best,
        results_sorted,
        calibrated_metrics,
        calibration_metrics,
        importance_summary,
        threshold_metrics,
        severity_conflict,
        operating_point,
        error_analysis,
        xgb_tuning,
        shap_summary,
        artifacts,
    )

    print("Model comparison:")
    for result in results_sorted:
        print(
            f"- {result.name}: AUC={result.roc_auc:.3f} "
            f"CV AUC={result.cv_auc:.3f}±{result.cv_std:.3f} "
            f"Brier={result.brier:.3f} LogLoss={result.log_loss:.3f}"
        )
    print(f"Selected model: {best.name}")
    print("Calibration method:", args.calibration)
    print(
        "XGBoost grid search best params:",
        xgb_tuning["best_params"],
        f"(CV AUC {xgb_tuning['best_cv_auc']:.3f}±{xgb_tuning['best_cv_std']:.3f})",
    )
    if shap_summary.get("available"):
        print(f"SHAP summary saved to {shap_summary['plot_path']}")
    else:
        print("SHAP summary skipped:", shap_summary.get("reason", "unknown"))
    print("Artifacts saved:", artifacts)
    print(
        "Operating point (flag-review threshold): "
        f"precision={operating_point['precision']:.3f} "
        f"recall={operating_point['recall']:.3f} "
        f"f1={operating_point['f1']:.3f}"
    )
    print(f"Outputs written to {args.outputs}")
    print(f"Evaluation report: {report_path}")


if __name__ == "__main__":
    main()
