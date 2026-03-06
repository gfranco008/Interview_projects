#!/usr/bin/env python3
"""
Synthetic alert dataset generator for TekStream assessment Task 1.

This module produces a 10k-row CSV of SOC alert events with realistic
distributions, correlated features, and engineered interaction effects.
It is intentionally deterministic given a random seed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -------------------------
# Configuration
# -------------------------

DEFAULT_CONFIG: dict = {
    "tactic_weights": [
        ["Reconnaissance", 0.19],
        ["Discovery", 0.18],
        ["Resource Development", 0.05],
        ["Initial Access", 0.08],
        ["Execution", 0.07],
        ["Persistence", 0.06],
        ["Privilege Escalation", 0.06],
        ["Defense Evasion", 0.06],
        ["Credential Access", 0.06],
        ["Lateral Movement", 0.04],
        ["Collection", 0.04],
        ["Command and Control", 0.05],
        ["Exfiltration", 0.03],
        ["Impact", 0.03],
    ],
    "tactic_severity": {
        "Reconnaissance": 1,
        "Resource Development": 1,
        "Discovery": 1,
        "Initial Access": 2,
        "Execution": 2,
        "Persistence": 2,
        "Privilege Escalation": 3,
        "Defense Evasion": 3,
        "Credential Access": 3,
        "Collection": 3,
        "Command and Control": 3,
        "Lateral Movement": 4,
        "Exfiltration": 4,
        "Impact": 4,
    },
    "tactic_techniques": {
        "Reconnaissance": ["T1595", "T1592", "T1590", "T1589"],
        "Resource Development": ["T1583", "T1587", "T1588", "T1585"],
        "Initial Access": ["T1190", "T1566", "T1078", "T1133"],
        "Execution": ["T1059", "T1203", "T1106", "T1047"],
        "Persistence": ["T1547", "T1053", "T1543", "T1037"],
        "Privilege Escalation": ["T1068", "T1078", "T1548", "T1134"],
        "Defense Evasion": ["T1070", "T1027", "T1036", "T1562"],
        "Credential Access": ["T1003", "T1110", "T1555", "T1556"],
        "Discovery": ["T1087", "T1018", "T1046", "T1082"],
        "Lateral Movement": ["T1021", "T1077", "T1091", "T1550"],
        "Collection": ["T1005", "T1114", "T1039", "T1025"],
        "Command and Control": ["T1071", "T1105", "T1219", "T1095"],
        "Exfiltration": ["T1041", "T1567", "T1020", "T1030"],
        "Impact": ["T1486", "T1499", "T1489", "T1491"],
    },
    "fp_range_by_severity": [
        [1, 0.65, 0.90],
        [2, 0.45, 0.70],
        [3, 0.25, 0.50],
        [4, 0.10, 0.30],
    ],
    "asset_type_weights": [
        ["workstation", 0.40],
        ["server", 0.25],
        ["cloud_vm", 0.18],
        ["iot", 0.12],
        ["domain_controller", 0.05],
    ],
    "asset_criticality_weights": {
        "domain_controller": [[8, 0.15], [9, 0.35], [10, 0.50]],
        "server": [[5, 0.10], [6, 0.20], [7, 0.30], [8, 0.25], [9, 0.10], [10, 0.05]],
        "cloud_vm": [[4, 0.10], [5, 0.20], [6, 0.30], [7, 0.25], [8, 0.15]],
        "workstation": [[2, 0.15], [3, 0.25], [4, 0.30], [5, 0.20], [6, 0.10]],
        "iot": [[1, 0.30], [2, 0.30], [3, 0.20], [4, 0.15], [5, 0.05]],
    },
    "user_privilege_weights": [
        ["standard", 0.70],
        ["admin", 0.20],
        ["service_account", 0.10],
    ],
    "vendor_severity_weights": [
        [1, 0.12],
        [2, 0.22],
        [3, 0.40],
        [4, 0.21],
        [5, 0.05],
    ],
    "rule_count": 50,
    "rule_fp_rate_buckets": [
        {"name": "high", "count": 10, "min": 0.82, "max": 0.95},
        {"name": "low", "count": 10, "min": 0.05, "max": 0.15},
        {"name": "mid", "count": 30, "min": 0.35, "max": 0.70},
    ],
    "source_pool_size": 250,
    "source_weight_sigma": 1.0,
    "source_burstiness_beta": [2.0, 4.0],
    "source_burstiness_spike_prob": 0.08,
    "source_burstiness_spike_add": [0.35, 0.70],
    "burstiness_weights": {
        "rule_fp": 0.45,
        "technique_fp": 0.25,
        "source": 0.20,
        "service_account": 0.10,
        "low_criticality": 0.05,
    },
    "count_scalars": {
        "base_offset": 1.0,
        "base_multiplier": 35.0,
        "severity_adjust_base": 1.1,
        "severity_adjust_step": 0.15,
        "lam24_min": 0.5,
        "lam1_min": 0.2,
        "lam1_ratio_base": 0.1,
        "lam1_ratio_multiplier": 0.6,
    },
    "tp_score_weights": {
        "severity": 1.1,
        "criticality": 0.9,
        "vendor": 0.5,
        "burst": -0.9,
        "rule_fp": -0.6,
        "technique_fp": -0.6,
        "service_account": -0.3,
        "interaction_sev_crit": 1.1,
        "interaction_burst_lowcrit": -0.9,
        "vendor_over_inherent": -0.4,
        "vendor5_bonus": 0.35,
    },
    "tp_score_noise": 0.2,
    "class_balance": {"true_positive": 0.15, "escalated": 0.10, "false_positive": 0.75},
    "vendor5_tp_target": 0.70,
    "ioc_match_rates": {
        "true_positive": 0.40,
        "false_positive": 0.03,
        "escalated": 0.12,
    },
    "fp_business_hour_bias": 0.75,
    "business_hours": [8, 17],
}

DERIVED_FEATURES = [
    {
        "name": "contextual_risk",
        "formula": "inherent_severity * asset_criticality",
        "justification": (
            "Combines intrinsic threat severity with asset impact to capture cross-axis risk."
        ),
        "math_justification": [
            "Let S=inherent_severity and C=asset_criticality; S*C is a multiplicative interaction. \n         If P(TP|S,C) grows super-multiplicatively, S*C increases class separability.",
        ],
        "expected_correlation": "Higher values should correlate with fewer false positives.",
    },
    {
        "name": "burst_index",
        "formula": "log1p(alert_count_1h) / log1p(alert_count_24h)",
        "justification": (
            "Measures short-term burstiness relative to daily volume, a proxy for noisy rules."
        ),
        "math_justification": [
            "Uses log1p to stabilize variance and bound extremes of count data. \n         The ratio is scale-stable and bounded in [0,1] when count_1h <= count_24h.",
        ],
        "expected_correlation": "Higher values should correlate with more false positives.",
    },
    {
        "name": "severity_gap",
        "formula": "inherent_severity - vendor_severity",
        "justification": (
            "Captures disagreement between intrinsic threat and vendor scoring."
        ),
        "math_justification": [
            "Signed residual e = S_inherent - S_vendor measures calibration bias. \n         Large negative e implies vendor inflation, which is predictive of FPs.",
        ],
        "expected_correlation": "Large negative gaps indicate likely false positives.",
    },
]

INTERACTION_EFFECTS = [
    {
        "name": "Severity x Criticality",
        "formula": "severity_norm * criticality_norm",
        "rationale": (
            "Late-stage tactics on high-criticality assets are more likely true positives."
        ),
    },
    {
        "name": "Burstiness x Low Criticality",
        "formula": "burst_index * (1 - criticality_norm)",
        "rationale": (
            "Bursty alerts on low-criticality assets are more likely false positives."
        ),
    },
]

REQUIRED_COLUMNS = [
    "alert_id",
    "timestamp",
    "source_rule",
    "mitre_tactic",
    "mitre_technique",
    "inherent_severity",
    "vendor_severity",
    "asset_type",
    "asset_criticality",
    "alert_count_1h",
    "alert_count_24h",
    "has_ioc_match",
    "user_privilege_level",
    "time_since_last_alert",
    "disposition",
]

DERIVED_COLUMNS = [
    "contextual_risk",
    "burst_index",
    "severity_gap",
]

OPTIONAL_EXTRA_COLUMNS = [
    "source_id",
    "alert_type",
    "detection_tool",
    "detection_vendor",
    "asset_id",
    "asset_hostname",
    "asset_ip",
    "user_id",
    "user_department",
    "process_name",
    "network_direction",
]

DETECTION_TOOLS = [
    "EDR",
    "IDS",
    "NDR",
    "SIEM",
    "Firewall",
    "Email Gateway",
    "CloudTrail",
]

DETECTION_VENDORS = [
    "CrowdStrike",
    "Microsoft",
    "Palo Alto Networks",
    "Splunk",
    "SentinelOne",
    "Cisco",
]

USER_DEPARTMENTS = [
    "Finance",
    "Engineering",
    "IT",
    "Security",
    "HR",
    "Operations",
]

PROCESS_NAMES = [
    "powershell.exe",
    "cmd.exe",
    "wmic.exe",
    "svchost.exe",
    "rundll32.exe",
    "python.exe",
    "curl.exe",
    "ssh",
    "bash",
]


def build_output_columns(include_extras: bool) -> List[str]:
    """Build the ordered CSV header based on selected options.

    Parameters
    ----------
    include_extras : bool
        Whether to include optional extra columns.

    Returns
    -------
    list[str]
        Ordered list of CSV column names.
    """
    if include_extras:
        return REQUIRED_COLUMNS + DERIVED_COLUMNS + OPTIONAL_EXTRA_COLUMNS
    return REQUIRED_COLUMNS + DERIVED_COLUMNS


# -------------------------
# Helpers
# -------------------------


def normalize_weights(
    items: Iterable[Tuple[Any, float]] | Dict[Any, float],
    value_cast=None,
) -> List[Tuple[Any, float]]:
    """Normalize weight pairs into a list of tuples.

    Parameters
    ----------
    items : Iterable
        Weight pairs as an iterable of (value, weight) or a mapping.
    value_cast : callable | None
        Optional cast applied to each value.

    Returns
    -------
    list[tuple[object, float]]
        Normalized (value, weight) pairs.
    """
    if isinstance(items, dict):
        iterable = items.items()
    else:
        iterable = items

    normalized = []
    for value, weight in iterable:
        if value_cast is not None:
            value = value_cast(value)
        normalized.append((value, float(weight)))
    return normalized


def normalize_fp_ranges(
    items: Dict[int, Tuple[float, float]] | List[List[float]],
) -> Dict[int, Tuple[float, float]]:
    """Normalize severity-to-FP-range mappings.

    Parameters
    ----------
    items : dict or list
        Either mapping of severity to (low, high) or a list of
        [severity, low, high] entries.

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping of severity to (low, high) FP rate ranges.
    """
    ranges: Dict[int, Tuple[float, float]] = {}
    if isinstance(items, dict):
        iterable = items.items()
        for sev, bounds in iterable:
            low, high = bounds
            ranges[int(sev)] = (float(low), float(high))
    else:
        for sev, low, high in items:
            ranges[int(sev)] = (float(low), float(high))
    return ranges


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a copy of the base dict.

    Parameters
    ----------
    base : dict
        Base configuration dictionary.
    overrides : dict
        Override values to apply.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[Path]) -> dict:
    """Load a JSON config override and merge with defaults.

    Parameters
    ----------
    config_path : Path | None
        Path to a JSON config override file.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    if config_path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a JSON object.")
    return deep_update(DEFAULT_CONFIG, raw)


def prepare_config(raw_config: dict) -> dict:
    """Normalize configuration for runtime use.

    Parameters
    ----------
    raw_config : dict
        Raw configuration dictionary.

    Returns
    -------
    dict
        Normalized configuration with prepared weight lists.
    """
    config = copy.deepcopy(raw_config)
    config["tactic_weights"] = normalize_weights(config["tactic_weights"])
    config["asset_type_weights"] = normalize_weights(config["asset_type_weights"])
    config["user_privilege_weights"] = normalize_weights(
        config["user_privilege_weights"]
    )
    config["vendor_severity_weights"] = normalize_weights(
        config["vendor_severity_weights"], value_cast=int
    )
    config["fp_range_by_severity"] = normalize_fp_ranges(config["fp_range_by_severity"])

    asset_weights = {}
    for asset, weights in config["asset_criticality_weights"].items():
        asset_weights[asset] = normalize_weights(weights, value_cast=int)
    config["asset_criticality_weights"] = asset_weights

    balance = config["class_balance"]
    if "false_positive" not in balance:
        balance["false_positive"] = (
            1.0 - balance["true_positive"] - balance["escalated"]
        )
    config["class_balance"] = balance
    return config


def validate_config(config: dict) -> None:
    """Validate configuration consistency and required keys.

    Parameters
    ----------
    config : dict
        Configuration dictionary (merged defaults + overrides).

    Raises
    ------
    ValueError
        If required keys are missing or inconsistent.
    """
    errors: List[str] = []

    def ensure_non_empty(key: str) -> None:
        value = config.get(key)
        if not value:
            errors.append(f"`{key}` must be a non-empty list.")

    ensure_non_empty("tactic_weights")
    ensure_non_empty("asset_type_weights")
    ensure_non_empty("user_privilege_weights")
    ensure_non_empty("vendor_severity_weights")
    ensure_non_empty("rule_fp_rate_buckets")

    if not config.get("asset_criticality_weights"):
        errors.append("`asset_criticality_weights` must be a non-empty mapping.")

    class_balance = config.get("class_balance", {})
    if not isinstance(class_balance, dict):
        errors.append("`class_balance` must be a mapping of class -> proportion.")
    else:
        if "true_positive" not in class_balance or "escalated" not in class_balance:
            errors.append("`class_balance` must include true_positive and escalated.")
        total = sum(
            float(v) for v in class_balance.values() if isinstance(v, (int, float))
        )
        if abs(total - 1.0) > 1e-3:
            errors.append(f"`class_balance` must sum to 1.0 (currently {total:.3f}).")

    fp_ranges = config.get("fp_range_by_severity", {})
    if isinstance(fp_ranges, dict):
        keys = {int(k) for k in fp_ranges.keys()}
    else:
        keys = {int(item[0]) for item in fp_ranges if len(item) >= 3}
    missing = {1, 2, 3, 4} - keys
    if missing:
        errors.append(
            "`fp_range_by_severity` must include severity keys 1-4 "
            f"(missing {sorted(missing)})."
        )

    asset_types = [item[0] for item in config.get("asset_type_weights", [])]
    crit_weights = config.get("asset_criticality_weights", {})
    for asset in asset_types:
        if asset not in crit_weights:
            errors.append(
                f"`asset_criticality_weights` must include asset_type '{asset}'."
            )
        elif not crit_weights[asset]:
            errors.append(f"`asset_criticality_weights['{asset}']` must be non-empty.")

    if errors:
        message = "Invalid config:\n" + "\n".join(f"- {err}" for err in errors)
        raise ValueError(message)


def weighted_choice(rng: random.Random, choices: Iterable[Tuple[Any, float]]) -> Any:
    """Sample a single value from weighted choices.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    choices : Iterable[tuple[object, float]]
        Iterable of (value, weight) pairs.

    Returns
    -------
    object
        The selected value.
    """
    choices_list = list(choices)
    if not choices_list:
        raise ValueError("choices must contain at least one item")

    total = sum(weight for _, weight in choices_list)
    r = rng.random() * total
    upto = 0.0
    for value, weight in choices_list:
        upto += weight
        if r <= upto:
            return value
    return choices_list[-1][0]


def random_private_ip(rng: random.Random) -> str:
    """Generate a random RFC1918 private IP address.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    str
        IPv4 address string.
    """
    block = rng.choice([10, 172, 192])
    if block == 10:
        return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    if block == 172:
        return f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def derive_alert_type(rule_fp_rate: float, technique_fp_rate: float) -> str:
    """Derive an alert type label from rule and technique FP rates.

    Parameters
    ----------
    rule_fp_rate : float
        Baseline FP rate for the source rule.
    technique_fp_rate : float
        Baseline FP rate for the technique.

    Returns
    -------
    str
        Alert type descriptor.
    """
    score = 0.6 * rule_fp_rate + 0.4 * technique_fp_rate
    if score >= 0.75:
        return "signature_broad"
    if score <= 0.20:
        return "behavioral"
    return "anomaly"


def derive_network_direction(tactic: str) -> str:
    """Map a MITRE tactic to a likely network direction.

    Parameters
    ----------
    tactic : str
        MITRE ATT&CK tactic.

    Returns
    -------
    str
        Network direction label.
    """
    if tactic in ("Exfiltration", "Command and Control"):
        return "outbound"
    if tactic in ("Initial Access", "Reconnaissance"):
        return "inbound"
    if tactic in ("Lateral Movement", "Discovery", "Credential Access"):
        return "lateral"
    return "internal"


def sample_process_name(rng: random.Random, tactic: str, asset_type: str) -> str:
    """Sample a process name with light conditioning on tactic or asset type.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    tactic : str
        MITRE ATT&CK tactic.
    asset_type : str
        Asset category.

    Returns
    -------
    str
        Process name string.
    """
    if tactic in ("Execution", "Persistence", "Privilege Escalation"):
        return rng.choice(["powershell.exe", "cmd.exe", "wmic.exe", "rundll32.exe"])
    if asset_type == "server":
        return rng.choice(["svchost.exe", "bash", "ssh", "python.exe"])
    return rng.choice(PROCESS_NAMES)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to the inclusive [low, high] range.

    Parameters
    ----------
    value : float
        Value to clamp.
    low : float
        Minimum allowed value.
    high : float
        Maximum allowed value.

    Returns
    -------
    float
        Clamped value.
    """
    return max(low, min(high, value))


def poisson(rng: random.Random, lam: float) -> int:
    """Sample a Poisson-distributed count using Knuth's algorithm.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    lam : float
        Poisson rate parameter (lambda).

    Returns
    -------
    int
        Sampled non-negative count.
    """
    # Good enough for the small-to-moderate rates used here.
    if lam <= 0:
        return 0
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1


def compute_burst_index(count_1h: int, count_24h: int) -> float:
    """Compute a normalized burstiness ratio using log scaling.

    Parameters
    ----------
    count_1h : int
        Alerts seen in the prior hour.
    count_24h : int
        Alerts seen in the prior 24 hours.

    Returns
    -------
    float
        Ratio in [0, 1] where higher means more bursty.
    """
    if count_24h <= 0:
        return 0.0
    denom = math.log1p(count_24h)
    if denom == 0:
        return 0.0
    return math.log1p(count_1h) / denom


def pearson(xs: List[float], ys: List[float]) -> float:
    """Compute a Pearson correlation coefficient.

    Parameters
    ----------
    xs : list[float]
        First numeric series.
    ys : list[float]
        Second numeric series.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


# -------------------------
# Build reference data
# -------------------------


def build_rules(
    rng: random.Random, config: dict
) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """Generate rule IDs with activity weights and baseline FP rates.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    config : dict
        Normalized configuration dictionary.

    Returns
    -------
    tuple[list[tuple[str, float]], dict[str, float]]
        Weighted rules list and per-rule FP rates.
    """
    rules_weights: List[Tuple[str, float]] = []
    rule_fp_rates: Dict[str, float] = {}

    rule_count = int(config["rule_count"])
    buckets = config["rule_fp_rate_buckets"]

    rule_index = 1
    for bucket in buckets:
        for _ in range(int(bucket["count"])):
            if rule_index > rule_count:
                break
            rule_id = f"RULE_{rule_index:03d}"
            fp_rate = rng.uniform(float(bucket["min"]), float(bucket["max"]))
            rule_fp_rates[rule_id] = fp_rate
            # Activity weight: a few noisy rules fire a lot more often.
            weight = rng.lognormvariate(0.0, 0.6)
            rules_weights.append((rule_id, weight))
            rule_index += 1

    # Fill any remaining rules using the last bucket range.
    if rule_index <= rule_count and buckets:
        fallback = buckets[-1]
        while rule_index <= rule_count:
            rule_id = f"RULE_{rule_index:03d}"
            fp_rate = rng.uniform(float(fallback["min"]), float(fallback["max"]))
            rule_fp_rates[rule_id] = fp_rate
            weight = rng.lognormvariate(0.0, 0.6)
            rules_weights.append((rule_id, weight))
            rule_index += 1

    return rules_weights, rule_fp_rates


def build_techniques(
    rng: random.Random, config: dict
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, float]]:
    """Generate technique weights and FP rates tied to tactic severity.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    config : dict
        Normalized configuration dictionary.

    Returns
    -------
    tuple[dict[str, list[tuple[str, float]]], dict[str, float]]
        Tactic-to-technique weights and per-technique FP rates.
    """
    tactic_to_tech_weights: Dict[str, List[Tuple[str, float]]] = {}
    technique_fp_rates: Dict[str, float] = {}
    for tactic, techniques in config["tactic_techniques"].items():
        severity = config["tactic_severity"][tactic]
        low, high = config["fp_range_by_severity"][severity]
        tactic_to_tech_weights[tactic] = []
        for tech_id in techniques:
            fp_rate = rng.uniform(low, high)
            technique_fp_rates[tech_id] = fp_rate
            tactic_to_tech_weights[tactic].append((tech_id, 1.0))
    return tactic_to_tech_weights, technique_fp_rates


def build_sources(
    rng: random.Random,
    config: dict,
    rule_weights: List[Tuple[str, float]],
    rule_fp_rates: Dict[str, float],
) -> List[Tuple[dict, float]]:
    """Create a pool of sources with stable asset and user context.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    config : dict
        Normalized configuration dictionary.
    rule_weights : list[tuple[str, float]]
        Weighted list of rule IDs used to assign a primary rule per source.
    rule_fp_rates : dict[str, float]
        Mapping of rule IDs to baseline FP rates.

    Returns
    -------
    list[tuple[dict, float]]
        Weighted list of source dictionaries with selection weights.
    """
    source_weights: List[Tuple[dict, float]] = []
    beta_a, beta_b = config["source_burstiness_beta"]
    spike_prob = config["source_burstiness_spike_prob"]
    spike_low, spike_high = config["source_burstiness_spike_add"]

    for i in range(1, int(config["source_pool_size"]) + 1):
        source_id = f"SRC_{i:04d}"
        asset_type = weighted_choice(rng, config["asset_type_weights"])
        asset_criticality = sample_asset_criticality(rng, asset_type, config)
        privilege = weighted_choice(rng, config["user_privilege_weights"])
        rule_id = weighted_choice(rng, rule_weights)
        rule_fp = rule_fp_rates[rule_id]

        source_burstiness = rng.betavariate(beta_a, beta_b)
        if rng.random() < spike_prob:
            source_burstiness = clamp(
                source_burstiness + rng.uniform(spike_low, spike_high), 0.0, 1.0
            )

        asset_prefix = {
            "domain_controller": "DC",
            "server": "SRV",
            "workstation": "WS",
            "cloud_vm": "VM",
            "iot": "IOT",
        }[asset_type]
        asset_id = f"AST_{i:04d}"
        asset_hostname = f"{asset_prefix}-{i:04d}"
        asset_ip = random_private_ip(rng)

        if privilege == "service_account":
            user_id = f"svc_{i:04d}"
        elif privilege == "admin":
            user_id = f"adm_{i:04d}"
        else:
            user_id = f"user_{i:04d}"

        user_department = rng.choice(USER_DEPARTMENTS)
        detection_tool = rng.choice(DETECTION_TOOLS)
        detection_vendor = rng.choice(DETECTION_VENDORS)

        weight = rng.lognormvariate(0.0, config["source_weight_sigma"])
        source_weights.append((
            {
                "source_id": source_id,
                "asset_type": asset_type,
                "asset_criticality": asset_criticality,
                "user_privilege_level": privilege,
                "source_rule": rule_id,
                "rule_fp_rate": rule_fp,
                "source_burstiness": source_burstiness,
                "asset_id": asset_id,
                "asset_hostname": asset_hostname,
                "asset_ip": asset_ip,
                "user_id": user_id,
                "user_department": user_department,
                "detection_tool": detection_tool,
                "detection_vendor": detection_vendor,
            },
            weight,
        ))
    return source_weights


# -------------------------
# Sampling functions
# -------------------------


def sample_vendor_severity(
    rng: random.Random, inherent_severity: int, config: dict
) -> int:
    """Sample vendor severity with intentional disagreement vs inherent severity.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    inherent_severity : int
        Inherent severity derived from MITRE tactic (1-4).
    config : dict
        Normalized configuration dictionary.

    Returns
    -------
    int
        Vendor-assigned severity on a 1-5 scale.
    """
    base = weighted_choice(rng, config["vendor_severity_weights"])
    roll = rng.random()
    if roll < 0.45:
        # push away from inherent severity to create disagreement
        shift = rng.choice([-2, -1, 1, 2])
        vendor = int(clamp(base + shift, 1, 5))
    elif roll < 0.75:
        # mild pull toward inherent severity
        vendor = int(clamp(round((base + inherent_severity) / 2), 1, 5))
    else:
        vendor = base
    return vendor


def sample_asset_criticality(rng: random.Random, asset_type: str, config: dict) -> int:
    """Sample asset criticality conditioned on asset type.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    asset_type : str
        Asset category (e.g., server, workstation).
    config : dict
        Normalized configuration dictionary.

    Returns
    -------
    int
        Asset criticality on a 1-10 scale.
    """
    return weighted_choice(rng, config["asset_criticality_weights"][asset_type])


def sample_counts(
    rng: random.Random,
    rule_fp: float,
    technique_fp: float,
    privilege: str,
    asset_criticality: int,
    inherent_severity: int,
    source_burstiness: float,
    config: dict,
) -> Tuple[int, int, float]:
    """Sample alert counts and return burst propensity for downstream features.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    rule_fp : float
        Baseline FP rate of the source rule.
    technique_fp : float
        Baseline FP rate of the technique.
    privilege : str
        User privilege level.
    asset_criticality : int
        Asset criticality on a 1-10 scale.
    inherent_severity : int
        Inherent severity on a 1-4 scale.
    source_burstiness : float
        Source-level burstiness factor.
    config : dict
        Normalized configuration dictionary.

    Returns
    -------
    tuple[int, int, float]
        (count_1h, count_24h, burst_propensity)
    """
    burst_weights = config["burstiness_weights"]
    burst_prop = (
        burst_weights["rule_fp"] * rule_fp
        + burst_weights["technique_fp"] * technique_fp
        + burst_weights["source"] * source_burstiness
    )
    if privilege == "service_account":
        burst_prop += burst_weights["service_account"]
    if asset_criticality <= 3:
        burst_prop += burst_weights["low_criticality"]
    burst_prop = clamp(burst_prop, 0.0, 1.0)

    scalars = config["count_scalars"]
    base = scalars["base_offset"] + scalars["base_multiplier"] * burst_prop
    severity_adjust = (
        scalars["severity_adjust_base"]
        - scalars["severity_adjust_step"] * inherent_severity
    )
    lam24 = max(scalars["lam24_min"], base * severity_adjust)
    lam1 = max(
        scalars["lam1_min"],
        lam24
        * (scalars["lam1_ratio_base"] + scalars["lam1_ratio_multiplier"] * burst_prop),
    )

    count_24h = poisson(rng, lam24)
    count_1h = poisson(rng, lam1)
    if count_1h > count_24h:
        count_1h = count_24h

    return count_1h, count_24h, burst_prop


def sample_time_since_last_alert(
    rng: random.Random, count_1h: int, count_24h: int, burst_prop: float
) -> int:
    """Sample time since last alert; bursty sources get shorter intervals.

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    count_1h : int
        Alerts seen in the prior hour.
    count_24h : int
        Alerts seen in the prior 24 hours.
    burst_prop : float
        Burst propensity in [0, 1].

    Returns
    -------
    int
        Seconds since last alert from this source.
    """
    if count_24h == 0:
        # quiet sources: hours to days
        return int(rng.uniform(6 * 3600, 7 * 24 * 3600))

    if count_1h == 0:
        base = 6 * 3600
    else:
        base = 3600 / (1 + count_1h / 2)

    scale = base * (1 - 0.6 * burst_prop)
    scale = clamp(scale, 60, 72 * 3600)
    return int(rng.expovariate(1 / scale))


def sample_timestamp(
    rng: random.Random,
    start: datetime,
    end: datetime,
    fp_business_hour_bias: bool,
    config: dict,
) -> datetime:
    """Sample timestamps with optional business-hours bias (for FPs).

    Parameters
    ----------
    rng : random.Random
        Random number generator instance.
    start : datetime
        Start of the time window.
    end : datetime
        End of the time window.
    fp_business_hour_bias : bool
        If True, concentrate more samples during business hours.
    config : dict
        Normalized configuration dictionary.

    Returns
    -------
    datetime
        Sampled timestamp.
    """
    total_seconds = int((end - start).total_seconds())
    if total_seconds <= 0:
        return start

    if not fp_business_hour_bias:
        offset = rng.uniform(0, total_seconds)
        return start + timedelta(seconds=offset)

    if rng.random() < config["fp_business_hour_bias"]:
        total_days = max(0, (end - start).days)
        day_offset = rng.randint(0, total_days)
        base_date = start + timedelta(days=day_offset)
        start_hour, end_hour = config["business_hours"]
        hour = rng.randint(start_hour, end_hour)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)

    offset = rng.uniform(0, total_seconds)
    return start + timedelta(seconds=offset)


def uuid7_from_sequence(base_ms: int, sequence: int) -> uuid.UUID:
    """Create a UUIDv7-like identifier using a base timestamp and sequence.

    Parameters
    ----------
    base_ms : int
        Base Unix timestamp in milliseconds.
    sequence : int
        Monotonic sequence number.

    Returns
    -------
    uuid.UUID
        RFC4122-compatible UUID with time-ordered properties.
    """
    unix_ms = (base_ms + sequence) & ((1 << 48) - 1)
    rand_a = sequence & 0x0FFF
    rand_b = (sequence >> 12) & ((1 << 62) - 1)

    uuid_int = (unix_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0x2 << 62) | rand_b
    return uuid.UUID(int=uuid_int)


def assign_dispositions(rows: List[dict], config: dict) -> None:
    """Assign dispositions by tp_score quantiles to enforce class balance.

    Parameters
    ----------
    rows : list[dict]
        Rows with computed tp_score values.
    config : dict
        Normalized configuration dictionary.
    """
    rows_sorted = sorted(rows, key=lambda r: r["tp_score"], reverse=True)
    balance = config["class_balance"]
    n_tp = int(len(rows) * balance["true_positive"])
    n_escalated = int(len(rows) * balance["escalated"])

    for idx, row in enumerate(rows_sorted):
        if idx < n_tp:
            row["disposition"] = "true_positive"
        elif idx < n_tp + n_escalated:
            row["disposition"] = "escalated"
        else:
            row["disposition"] = "false_positive"


def enforce_vendor5_tp_rate(rows: List[dict], target_rate: float) -> None:
    """Promote vendor_severity=5 alerts to meet a TP rate target.

    Parameters
    ----------
    rows : list[dict]
        Dataset rows with assigned dispositions.
    target_rate : float
        Desired true positive rate within vendor_severity=5.
    """
    vendor5_rows = [r for r in rows if r["vendor_severity"] == 5]
    if not vendor5_rows:
        return

    tp_vendor5 = [r for r in vendor5_rows if r["disposition"] == "true_positive"]
    current_rate = len(tp_vendor5) / len(vendor5_rows)
    if current_rate >= target_rate:
        return

    promotable = sorted(
        (r for r in vendor5_rows if r["disposition"] != "true_positive"),
        key=lambda r: r["tp_score"],
        reverse=True,
    )
    demotable = sorted(
        (
            r
            for r in rows
            if r["disposition"] == "true_positive" and r["vendor_severity"] != 5
        ),
        key=lambda r: r["tp_score"],
    )

    for promote_row in promotable:
        if not demotable:
            break
        current_rate = len(tp_vendor5) / len(vendor5_rows)
        if current_rate >= target_rate:
            break

        demote_row = demotable.pop(0)
        old_label = promote_row["disposition"]
        promote_row["disposition"] = "true_positive"
        demote_row["disposition"] = old_label
        tp_vendor5.append(promote_row)


# -------------------------
# Generation
# -------------------------


def generate_rows(
    n_rows: int,
    seed: int,
    end_timestamp: Optional[datetime],
    config: dict,
    include_extras: bool,
) -> List[dict]:
    """Generate synthetic alerts and return them as a list of row dicts.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    seed : int
        RNG seed for deterministic output.
    end_timestamp : datetime | None
        End timestamp for the 90-day window (defaults to now).
    config : dict
        Normalized configuration dictionary.
    include_extras : bool
        Whether to populate optional extra columns.

    Returns
    -------
    list[dict]
        List of row dictionaries containing all dataset columns.

    Notes
    -----
    Dispositions are assigned by tp_score quantiles to enforce the target
    class balance (15% true_positive, 10% escalated, 75% false_positive).
    """
    rng = random.Random(seed)
    rule_weights, rule_fp_rates = build_rules(rng, config)
    tactic_to_tech_weights, technique_fp_rates = build_techniques(rng, config)
    source_weights = build_sources(rng, config, rule_weights, rule_fp_rates)

    now = end_timestamp or datetime.now().replace(microsecond=0)
    start = now - timedelta(days=90)
    base_uuid_ms = int(start.timestamp() * 1000)

    rows: List[dict] = []
    for i in range(n_rows):
        source = weighted_choice(rng, source_weights)

        tactic = weighted_choice(rng, config["tactic_weights"])
        technique = weighted_choice(rng, tactic_to_tech_weights[tactic])
        inherent_severity = config["tactic_severity"][tactic]

        vendor_severity = sample_vendor_severity(rng, inherent_severity, config)
        technique_fp = technique_fp_rates[technique]

        alert_count_1h, alert_count_24h, burst_prop = sample_counts(
            rng,
            source["rule_fp_rate"],
            technique_fp,
            source["user_privilege_level"],
            source["asset_criticality"],
            inherent_severity,
            source["source_burstiness"],
            config,
        )
        time_since_last = sample_time_since_last_alert(
            rng, alert_count_1h, alert_count_24h, burst_prop
        )

        burst_index = compute_burst_index(alert_count_1h, alert_count_24h)
        severity_gap = inherent_severity - vendor_severity
        contextual_risk = inherent_severity * source["asset_criticality"]

        severity_norm = inherent_severity / 4
        criticality_norm = source["asset_criticality"] / 10
        vendor_norm = vendor_severity / 5

        tp_weights = config["tp_score_weights"]
        tp_score = (
            tp_weights["severity"] * severity_norm
            + tp_weights["criticality"] * criticality_norm
            + tp_weights["vendor"] * vendor_norm
            + tp_weights["burst"] * burst_index
            + tp_weights["rule_fp"] * source["rule_fp_rate"]
            + tp_weights["technique_fp"] * technique_fp
            + tp_weights["service_account"]
            * (1 if source["user_privilege_level"] == "service_account" else 0)
        )

        tp_score += tp_weights["interaction_sev_crit"] * (
            severity_norm * criticality_norm
        )
        tp_score += tp_weights["interaction_burst_lowcrit"] * (
            burst_index * (1 - criticality_norm)
        )
        tp_score += tp_weights["vendor_over_inherent"] * max(
            0.0, vendor_norm - severity_norm
        )
        tp_score += tp_weights["vendor5_bonus"] * (1 if vendor_severity == 5 else 0)
        tp_score += rng.normalvariate(0, config["tp_score_noise"])

        row = {
            "row_index": i,
            "source_id": source["source_id"],
            "mitre_tactic": tactic,
            "mitre_technique": technique,
            "inherent_severity": inherent_severity,
            "vendor_severity": vendor_severity,
            "asset_type": source["asset_type"],
            "asset_criticality": source["asset_criticality"],
            "source_rule": source["source_rule"],
            "user_privilege_level": source["user_privilege_level"],
            "alert_count_1h": alert_count_1h,
            "alert_count_24h": alert_count_24h,
            "time_since_last_alert": time_since_last,
            "technique_fp_rate": technique_fp,
            "rule_fp_rate": source["rule_fp_rate"],
            "burst_index": burst_index,
            "severity_gap": severity_gap,
            "contextual_risk": contextual_risk,
            "tp_score": tp_score,
        }

        if include_extras:
            row.update({
                "alert_type": derive_alert_type(source["rule_fp_rate"], technique_fp),
                "detection_tool": source["detection_tool"],
                "detection_vendor": source["detection_vendor"],
                "asset_id": source["asset_id"],
                "asset_hostname": source["asset_hostname"],
                "asset_ip": source["asset_ip"],
                "user_id": source["user_id"],
                "user_department": source["user_department"],
                "process_name": sample_process_name(rng, tactic, source["asset_type"]),
                "network_direction": derive_network_direction(tactic),
            })

        rows.append(row)

    assign_dispositions(rows, config)
    enforce_vendor5_tp_rate(rows, config["vendor5_tp_target"])

    for row in rows:
        disposition = row["disposition"]
        ioc_prob = config["ioc_match_rates"][disposition]
        fp_bias = disposition == "false_positive"

        row["timestamp"] = sample_timestamp(rng, start, now, fp_bias, config).isoformat(
            sep=" "
        )
        row["has_ioc_match"] = rng.random() < ioc_prob
        row["alert_id"] = str(uuid7_from_sequence(base_uuid_ms, row["row_index"]))

    return rows


# -------------------------
# Summary
# -------------------------


def validate_rows(rows: List[dict], required_columns: List[str]) -> List[str]:
    """Validate required columns exist across all rows.

    Parameters
    ----------
    rows : list[dict]
        Dataset rows to validate.
    required_columns : list[str]
        Columns required by the spec.

    Returns
    -------
    list[str]
        Validation warning messages (empty if none).
    """
    warnings: List[str] = []
    if not rows:
        return ["No rows generated."]

    missing = set()
    for row in rows:
        for column in required_columns:
            if column not in row:
                missing.add(column)

    if missing:
        warnings.append(f"Missing required columns: {', '.join(sorted(missing))}.")
    return warnings


def validate_distributions(
    rows: List[dict], config: dict, expected_rows: int
) -> List[str]:
    """Validate dataset distributions against key requirements.

    Parameters
    ----------
    rows : list[dict]
        Dataset rows to validate.
    config : dict
        Normalized configuration dictionary.
    expected_rows : int
        Expected row count (from CLI).

    Returns
    -------
    list[str]
        Validation warning messages (empty if none).
    """
    warnings: List[str] = []
    n = len(rows)
    if n != expected_rows:
        warnings.append(f"Row count {n} does not match expected {expected_rows}.")

    counts = Counter(r["disposition"] for r in rows)
    tolerance = 0.02
    for label, target in config["class_balance"].items():
        actual = counts.get(label, 0) / max(1, n)
        if abs(actual - target) > tolerance:
            warnings.append(
                f"Class balance for {label} is {actual:.2%}, expected {target:.2%}."
            )

    for label, expected in config["ioc_match_rates"].items():
        subset = [r for r in rows if r["disposition"] == label]
        if not subset:
            continue
        actual = sum(1 for r in subset if r["has_ioc_match"]) / len(subset)
        if abs(actual - expected) > 0.05:
            warnings.append(
                f"IOC match rate for {label} is {actual:.2%}, expected {expected:.2%}."
            )

    tactic_counts = Counter(r["mitre_tactic"] for r in rows)
    top_tactics = [t for t, _ in tactic_counts.most_common(3)]
    if "Reconnaissance" not in top_tactics or "Discovery" not in top_tactics:
        warnings.append("Reconnaissance and Discovery are not among the top 3 tactics.")

    impact_rate = tactic_counts.get("Impact", 0) / max(1, n)
    exfil_rate = tactic_counts.get("Exfiltration", 0) / max(1, n)
    if impact_rate > 0.06:
        warnings.append("Impact tactic is not rare enough (over 6%).")
    if exfil_rate > 0.06:
        warnings.append("Exfiltration tactic is not rare enough (over 6%).")

    vendor5_rows = [r for r in rows if r["vendor_severity"] == 5]
    if vendor5_rows:
        vendor5_rate = len(vendor5_rows) / n
        tp_rate = sum(
            1 for r in vendor5_rows if r["disposition"] == "true_positive"
        ) / len(vendor5_rows)
        if vendor5_rate > 0.10:
            warnings.append("Vendor severity 5 is not rare enough (over 10%).")
        if tp_rate < config["vendor5_tp_target"]:
            warnings.append(
                f"Vendor severity 5 TP rate is {tp_rate:.2%}, below target."
            )
    else:
        warnings.append("No vendor severity 5 alerts generated.")

    rule_fp_rates = {
        r["source_rule"]: r.get("rule_fp_rate")
        for r in rows
        if r.get("rule_fp_rate") is not None
    }
    high_rules = sum(1 for v in rule_fp_rates.values() if v >= 0.80)
    low_rules = sum(1 for v in rule_fp_rates.values() if v <= 0.10)
    if high_rules < 8:
        warnings.append("Fewer than 8 rules have FP rates above 80%.")
    if low_rules < 8:
        warnings.append("Fewer than 8 rules have FP rates below 10%.")

    crit_values = [r["asset_criticality"] for r in rows]
    if crit_values:
        if min(crit_values) < 1 or max(crit_values) > 10:
            warnings.append("Asset criticality is outside the 1-10 range.")

    return warnings


def build_summary(rows: List[dict], config: dict, warnings: List[str]) -> str:
    """Create a brief statistical summary of key dataset properties.

    Parameters
    ----------
    rows : list[dict]
        Dataset rows as generated by `generate_rows`.
    config : dict
        Normalized configuration dictionary.
    warnings : list[str]
        Validation warnings to include in the summary.

    Returns
    -------
    str
        Multi-line summary string.
    """
    n = len(rows)
    counts = Counter(r["disposition"] for r in rows)

    def pct(count: int) -> float:
        return 100.0 * count / n if n else 0.0

    tactic_counts = Counter(r["mitre_tactic"] for r in rows)
    top_tactics = tactic_counts.most_common(5)

    inherent = [r["inherent_severity"] for r in rows]
    vendor = [r["vendor_severity"] for r in rows]
    burst = [r["burst_index"] for r in rows]
    is_fp = [1 if r["disposition"] == "false_positive" else 0 for r in rows]

    corr_inherent_vendor = pearson(inherent, vendor)
    corr_burst_fp = pearson(burst, is_fp)

    disagreement = [
        1 for r in rows if abs(r["inherent_severity"] - r["vendor_severity"]) >= 2
    ]
    disagreement_rate = 100.0 * sum(disagreement) / n if n else 0.0

    high_sev_high_crit = [
        r for r in rows if r["inherent_severity"] >= 3 and r["asset_criticality"] >= 8
    ]
    high_sev_tp_rate = (
        100.0
        * sum(1 for r in high_sev_high_crit if r["disposition"] == "true_positive")
        / max(1, len(high_sev_high_crit))
    )

    bursty_low_crit = [
        r for r in rows if r["burst_index"] >= 0.7 and r["asset_criticality"] <= 3
    ]
    bursty_fp_rate = (
        100.0
        * sum(1 for r in bursty_low_crit if r["disposition"] == "false_positive")
        / max(1, len(bursty_low_crit))
    )

    vendor5_rows = [r for r in rows if r["vendor_severity"] == 5]
    vendor5_rate = pct(len(vendor5_rows))
    vendor5_tp_rate = (
        100.0
        * sum(1 for r in vendor5_rows if r["disposition"] == "true_positive")
        / max(1, len(vendor5_rows))
    )

    crit_values = [r["asset_criticality"] for r in rows]
    crit_min = min(crit_values) if crit_values else None
    crit_max = max(crit_values) if crit_values else None

    lines = []
    lines.append(f"Dataset size: {n}")
    lines.append(
        "Class balance: "
        f"false_positive {pct(counts.get('false_positive', 0)):.1f}% "
        f"({counts.get('false_positive', 0)}), "
        f"true_positive {pct(counts.get('true_positive', 0)):.1f}% "
        f"({counts.get('true_positive', 0)}), "
        f"escalated {pct(counts.get('escalated', 0)):.1f}% "
        f"({counts.get('escalated', 0)})"
    )
    lines.append(
        "Top tactics: " + ", ".join(f"{t} {pct(c):.1f}%" for t, c in top_tactics)
    )
    lines.append(
        f"Severity disagreement |gap|>=2: {disagreement_rate:.1f}% (vendor vs inherent)"
    )
    lines.append(
        "Correlation: inherent vs vendor severity r="
        f"{corr_inherent_vendor:.2f}; burst_index vs FP label r={corr_burst_fp:.2f}"
    )
    lines.append(
        f"Vendor severity=5 rate: {vendor5_rate:.1f}% "
        f"(TP within vendor=5: {vendor5_tp_rate:.1f}%)"
    )
    if crit_min is not None and crit_max is not None:
        lines.append(f"Asset criticality range: {crit_min} to {crit_max}")

    lines.append("")
    lines.append("Interaction effects:")
    lines.append(f"1. {INTERACTION_EFFECTS[0]['name']}")
    lines.append(f"   Formula: {INTERACTION_EFFECTS[0]['formula']}")
    lines.append(f"   Rationale: {INTERACTION_EFFECTS[0]['rationale']}")
    lines.append(
        f"   Empirical check: high/high TP rate {high_sev_tp_rate:.1f}% "
        f"vs baseline {pct(counts.get('true_positive', 0)):.1f}%"
    )
    lines.append(f"2. {INTERACTION_EFFECTS[1]['name']}")
    lines.append(f"   Formula: {INTERACTION_EFFECTS[1]['formula']}")
    lines.append(f"   Rationale: {INTERACTION_EFFECTS[1]['rationale']}")
    lines.append(
        f"   Empirical check: bursty/low-criticality FP rate {bursty_fp_rate:.1f}% "
        f"vs baseline {pct(counts.get('false_positive', 0)):.1f}%"
    )

    lines.append("")
    lines.append("Derived features (Task 1B):")
    for idx, feature in enumerate(DERIVED_FEATURES, start=1):
        lines.append(f"{idx}. {feature['name']} = {feature['formula']}")
        lines.append(f"   Justification: {feature['justification']}")
        for math_line in feature.get("math_justification", []):
            lines.append(f"   Mathematical justification: {math_line}")
        lines.append(f"   Expected correlation: {feature['expected_correlation']}")

    if warnings:
        lines.append("")
        lines.append("Validation warnings:")
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines)


# -------------------------
# Main
# -------------------------


def main() -> None:
    """CLI entrypoint to generate the dataset and summary.

    Notes
    -----
    The generated CSV and summary are written into the `outputs/` directory
    relative to this script unless overridden via CLI arguments.
    Use `--extra-columns` to include optional SOC context fields.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic alert dataset")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file that overrides defaults",
    )
    parser.add_argument(
        "--extra-columns",
        action="store_true",
        help="Include optional extra SOC columns in the output CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "alerts_dataset.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "summary.txt",
    )
    parser.add_argument(
        "--end-timestamp",
        type=str,
        default=None,
        help="ISO timestamp to anchor the 90-day window (default: now)",
    )
    args = parser.parse_args()

    raw_config = load_config(args.config)
    validate_config(raw_config)
    config = prepare_config(raw_config)

    end_ts = None
    if args.end_timestamp:
        end_ts = datetime.fromisoformat(args.end_timestamp)

    rows = generate_rows(args.rows, args.seed, end_ts, config, args.extra_columns)

    output_columns = build_output_columns(args.extra_columns)

    warnings = []
    warnings.extend(validate_rows(rows, output_columns))
    warnings.extend(validate_distributions(rows, config, args.rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in output_columns})

    summary = build_summary(rows, config, warnings)
    args.summary.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nWrote dataset to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
