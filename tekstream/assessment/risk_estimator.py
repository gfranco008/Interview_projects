#!/usr/bin/env python3
"""Estimate a rolling risk state from alert data (Task 3)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    half_life_hours: float
    progression_window_hours: float
    bucket: str
    max_alert_risk: float
    gamma: float
    w_inherent: float
    w_vendor: float
    w_context: float
    w_behavior: float
    w_disposition: float


KILL_CHAIN_STAGE = {
    "Reconnaissance": 1,
    "Discovery": 1,
    "Resource Development": 1,
    "Initial Access": 2,
    "Execution": 3,
    "Persistence": 3,
    "Defense Evasion": 3,
    "Credential Access": 4,
    "Privilege Escalation": 4,
    "Lateral Movement": 5,
    "Command and Control": 5,
    "Collection": 5,
    "Exfiltration": 6,
    "Impact": 6,
}


def clamp01(value: float) -> float:
    """Clamp a numeric value to the [0, 1] range."""
    return float(min(1.0, max(0.0, value)))


def load_alerts(path: Path) -> pd.DataFrame:
    """Load alert data and parse timestamps."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    return df


def ensure_customer_id(df: pd.DataFrame, num_customers: int) -> pd.DataFrame:
    """Ensure a customer_id column exists by assigning synthetic customers."""
    if "customer_id" in df.columns:
        return df
    df = df.copy()
    key = (
        df.get("source_rule", pd.Series(["unknown"] * len(df))).astype(str)
        + "|"
        + df.get("asset_type", pd.Series(["unknown"] * len(df))).astype(str)
    )
    codes, _ = pd.factorize(key)
    df["customer_id"] = (codes % max(1, num_customers)) + 1
    df["customer_id"] = df["customer_id"].map(lambda value: f"customer_{value}")
    return df


def compute_alert_impact(row: pd.Series, config: RiskConfig, progression_ratio: float) -> float:
    """Compute the normalized alert contribution (Δ) for the risk update."""
    inherent = clamp01(float(row.get("inherent_severity", 0) or 0) / 4.0)
    vendor = clamp01(float(row.get("vendor_severity", inherent) or inherent) / 5.0)
    criticality = clamp01(float(row.get("asset_criticality", 0) or 0) / 10.0)
    burst_norm = clamp01(float(row.get("burst_index", 0) or 0))
    ioc = 1.0 if bool(row.get("has_ioc_match", 0)) else 0.0

    disposition = str(row.get("disposition", "")).lower()
    if disposition == "false_positive":
        disposition_score = 0.1
    elif disposition == "escalated":
        disposition_score = 0.6
    elif disposition == "true_positive":
        disposition_score = 1.0
    else:
        disposition_score = 0.5

    behavior = 0.5 * burst_norm + 0.5 * ioc

    base = (
        config.w_inherent * inherent
        + config.w_vendor * vendor
        + config.w_context * criticality
        + config.w_behavior * behavior
        + config.w_disposition * disposition_score
    )
    base = float(min(config.max_alert_risk, max(0.0, base)))
    if progression_ratio > 0:
        base *= 1.0 + config.gamma * progression_ratio
        base = float(min(config.max_alert_risk, base))
    return base


def compute_kill_chain_bonus(
    recent_stages: List[Tuple[pd.Timestamp, int]],
    current_time: pd.Timestamp,
    stage: int | None,
    window_hours: float,
) -> Tuple[float, List[Tuple[pd.Timestamp, int]]]:
    """Compute a lightweight kill-chain progression ratio within a rolling window."""
    if stage is None:
        return 0.0, recent_stages

    windowed = [
        (timestamp, value)
        for timestamp, value in recent_stages
        if (current_time - timestamp).total_seconds() / 3600.0 <= window_hours
    ]
    windowed.append((current_time, stage))

    ordered = []
    for _, value in sorted(windowed, key=lambda item: item[0]):
        if value not in ordered:
            ordered.append(value)

    if len(ordered) < 2:
        return 0.0, windowed

    pairs = 0
    max_pairs = len(ordered) * (len(ordered) - 1) / 2
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if ordered[j] > ordered[i]:
                pairs += 1
    ratio = pairs / max_pairs if max_pairs > 0 else 0.0
    return float(ratio), windowed


def apply_decay(current_risk: float, delta_hours: float, decay_rate: float) -> float:
    """Apply exponential decay to the current risk based on elapsed hours."""
    if delta_hours <= 0:
        return current_risk
    return float(current_risk * np.exp(-decay_rate * delta_hours))


def update_risk_state(decayed_risk: float, delta_risk: float) -> float:
    """Update bounded risk state with the new alert contribution."""
    updated = 1.0 - (1.0 - decayed_risk) * (1.0 - delta_risk)
    return clamp01(updated)


def estimate_risk_over_time(df: pd.DataFrame, config: RiskConfig) -> pd.DataFrame:
    """Estimate rolling risk scores for each customer over time."""
    decay_rate = log(2) / config.half_life_hours
    records = []

    for customer_id, group in df.groupby("customer_id"):
        group = group.sort_values("timestamp")
        last_time = None
        current_risk = 0.0
        recent_stages: List[Tuple[pd.Timestamp, int]] = []

        for idx, row in group.iterrows():
            if last_time is None:
                delta_hours = 0.0
            else:
                delta_hours = (row["timestamp"] - last_time).total_seconds() / 3600.0

            decayed = apply_decay(current_risk, delta_hours, decay_rate)
            tactic = str(row.get("mitre_tactic", ""))
            stage = KILL_CHAIN_STAGE.get(tactic, None)
            progression_ratio, recent_stages = compute_kill_chain_bonus(
                recent_stages, row["timestamp"], stage, config.progression_window_hours
            )
            delta = compute_alert_impact(row, config, progression_ratio)
            current_risk = update_risk_state(decayed, delta)

            records.append(
                {
                    "index": idx,
                    "risk_score": current_risk,
                    "decayed_risk": decayed,
                    "delta_risk": delta,
                    "progression_ratio": progression_ratio,
                }
            )
            last_time = row["timestamp"]

    metrics = pd.DataFrame(records).set_index("index")
    return df.join(metrics)


def pick_customer(df: pd.DataFrame, customer_id: str | None) -> str:
    """Pick a customer to plot, defaulting to the highest-volume one."""
    customers = df["customer_id"].astype(str)
    if customer_id and customer_id in customers.unique():
        return customer_id
    counts = customers.value_counts()
    return str(counts.index[0])


def collect_annotations(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, float, str]]:
    """Collect labeled annotations for elevated-risk and false-alarm periods."""
    annotations: List[Tuple[pd.Timestamp, float, str]] = []
    if df.empty:
        return annotations
    fallback_numeric = pd.Series([0] * len(df), index=df.index)
    fallback_text = pd.Series([""] * len(df), index=df.index)

    risk_threshold = max(0.5, float(df["risk_score"].quantile(0.75)))
    disposition_series = df.get("disposition", fallback_text).astype(str).str.lower()
    tp_escalated_mask = disposition_series.isin(["true_positive", "escalated"])
    elevated_mask = (
        (df["risk_score"] >= risk_threshold)
        & (
            tp_escalated_mask
            | (df["progression_ratio"] >= 0.3)
        )
    )
    if elevated_mask.any():
        row = df[elevated_mask].nlargest(1, "risk_score").iloc[0]
        if row["progression_ratio"] >= 0.3:
            label = "Elevated risk period: progression into later kill-chain stages"
        else:
            label = "Elevated risk period: high-severity true positives"
        annotations.append((row["timestamp"], float(row["risk_score"]), label))
    elif tp_escalated_mask.any():
        row = df[tp_escalated_mask].nlargest(1, "risk_score").iloc[0]
        annotations.append(
            (
                row["timestamp"],
                float(row["risk_score"]),
                "Elevated risk period: true positives/escalations present",
            )
        )
    else:
        row = df.nlargest(1, "risk_score").iloc[0]
        annotations.append((row["timestamp"], float(row["risk_score"]), "Elevated risk period"))

    burst_candidates = df[
        (df.get("asset_criticality", fallback_numeric) <= 3)
        & (df.get("burst_index", fallback_numeric) >= 0.75)
        & (disposition_series == "false_positive")
    ]
    if not burst_candidates.empty:
        row = burst_candidates.nlargest(1, "burst_index").iloc[0]
        annotations.append(
            (
                row["timestamp"],
                float(row["risk_score"]),
                "False alarm period: bursty low-criticality FP cluster",
            )
        )

    progression = df[df["progression_ratio"] >= 0.5]
    if not progression.empty:
        row = progression.nlargest(1, "progression_ratio").iloc[0]
        too_close = False
        for ts, _, _ in annotations:
            if abs((row["timestamp"] - ts).total_seconds()) < 6 * 3600:
                too_close = True
                break
        if not too_close:
            annotations.append(
                (
                    row["timestamp"],
                    float(row["risk_score"]),
                    "Kill-chain progression bonus",
                )
            )

    return annotations


def summarize_elevated_period(df: pd.DataFrame) -> str:
    """Summarize the strongest elevated-risk period for console output."""
    if df.empty:
        return "Elevated-risk period: none (empty dataset)."

    risk_threshold = max(0.5, float(df["risk_score"].quantile(0.75)))
    disposition_series = df.get("disposition", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    tp_escalated_mask = disposition_series.isin(["true_positive", "escalated"])
    elevated_mask = (
        (df["risk_score"] >= risk_threshold)
        & (
            tp_escalated_mask
            | (df["progression_ratio"] >= 0.3)
        )
    )

    if elevated_mask.any():
        row = df[elevated_mask].nlargest(1, "risk_score").iloc[0]
        if row["progression_ratio"] >= 0.3:
            driver = "kill-chain progression + true positives/escalations"
        else:
            driver = "true positives/escalations"
        return (
            "Elevated-risk period: "
            f"{row['timestamp']} (risk={row['risk_score']:.2f}, driver={driver})."
        )

    if tp_escalated_mask.any():
        row = df[tp_escalated_mask].nlargest(1, "risk_score").iloc[0]
        return (
            "Elevated-risk period: "
            f"{row['timestamp']} (risk={row['risk_score']:.2f}, driver=true positives/escalations)."
        )

    row = df.nlargest(1, "risk_score").iloc[0]
    return (
        "Elevated-risk period: "
        f"{row['timestamp']} (risk={row['risk_score']:.2f}, driver=highest observed risk)."
    )


def plot_risk_timeline(df: pd.DataFrame, output_path: Path, bucket: str) -> None:
    """Plot the risk timeline with annotated events."""
    series = df.set_index("timestamp")["risk_score"].resample(bucket).last().ffill()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color="#2f3e46", linewidth=2)

    color_map = {
        "true_positive": "#2a9d8f",
        "escalated": "#e9c46a",
        "false_positive": "#e76f51",
    }
    if "disposition" in df.columns:
        for disposition, color in color_map.items():
            subset = df[df["disposition"].astype(str).str.lower() == disposition]
            if subset.empty:
                continue
            sizes = 30 + 220 * subset.get("delta_risk", pd.Series([0] * len(subset), index=subset.index))
            sizes = np.clip(sizes, 20, 260)
            ax.scatter(
                subset["timestamp"],
                subset["risk_score"],
                s=sizes,
                alpha=0.6,
                color=color,
                label=disposition.replace("_", " "),
            )

    for ts, value, label in collect_annotations(df):
        ax.annotate(
            label,
            (ts, value),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="#1f2937",
            arrowprops=dict(arrowstyle="->", color="#1f2937", linewidth=0.6),
        )

    ax.set_title("Risk Timeline (EWMA + Severity Weighting + Kill Chain Bonus)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Risk score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_decay_vs_delta(df: pd.DataFrame, output_path: Path) -> None:
    """Plot decayed risk versus alert contribution (Δ)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {
        "true_positive": "#2a9d8f",
        "escalated": "#e9c46a",
        "false_positive": "#e76f51",
    }
    disposition_series = df.get("disposition", pd.Series(["unknown"] * len(df), index=df.index))
    for disposition, color in color_map.items():
        subset = df[disposition_series.astype(str).str.lower() == disposition]
        if subset.empty:
            continue
        ax.scatter(
            subset["decayed_risk"],
            subset["delta_risk"],
            s=28,
            alpha=0.6,
            color=color,
            label=disposition.replace("_", " "),
        )

    ax.set_title("Decayed Risk vs. Alert Contribution")
    ax.set_xlabel("Decayed risk")
    ax.set_ylabel("Alert contribution (Δ)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint for estimating risk and generating plots."""
    parser = argparse.ArgumentParser(description="Estimate rolling risk state")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "alerts_dataset.csv",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--half-life-hours", type=float, default=24.0)
    parser.add_argument("--progression-window-hours", type=float, default=24.0)
    parser.add_argument("--bucket", type=str, default="D")
    parser.add_argument("--customer-id", type=str, default="")
    parser.add_argument("--num-customers", type=int, default=8)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset}. Run generate_dataset.py first."
        )

    df = load_alerts(args.dataset)
    df = ensure_customer_id(df, args.num_customers)

    config = RiskConfig(
        half_life_hours=args.half_life_hours,
        progression_window_hours=args.progression_window_hours,
        bucket=args.bucket,
        max_alert_risk=0.95,
        gamma=0.4,
        w_inherent=0.45,
        w_vendor=0.10,
        w_context=0.20,
        w_behavior=0.15,
        w_disposition=0.10,
    )

    scored = estimate_risk_over_time(df, config)
    target_customer = pick_customer(scored, args.customer_id or None)
    scored = scored[scored["customer_id"] == target_customer].copy()

    args.outputs.mkdir(parents=True, exist_ok=True)
    plot_risk_timeline(scored, args.outputs / "risk_timeline.png", config.bucket)
    plot_decay_vs_delta(scored, args.outputs / "risk_decay_vs_delta.png")

    print(f"Customer plotted: {target_customer}")
    print(f"Risk timeline saved to {args.outputs / 'risk_timeline.png'}")
    print(f"Risk decay vs delta plot saved to {args.outputs / 'risk_decay_vs_delta.png'}")
    print(summarize_elevated_period(scored))


if __name__ == "__main__":
    main()
