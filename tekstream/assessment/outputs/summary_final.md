# Dataset Summary — MDR Alert Generator

This document describes the synthetic dataset produced by `generate_dataset.py` for the TekStream MDR assessment. It covers dataset composition, class balance, feature distributions, interaction effects, derived features, and validation notes.

---

## 1. Dataset Overview

| Property | Value |
| --- | --- |
| Total alerts | 10,000 |
| Generation method | Deterministic synthetic generator with configurable seed |
| Output file | `outputs/alerts_dataset.csv` |

The dataset is designed to reflect realistic MDR alert volume and noise characteristics. It is intentionally imbalanced, skewed toward early-stage MITRE tactics, and includes two explicit interaction effects that the ML model should learn to rediscover.

---

## 2. Class Balance

| Disposition | Count | Percentage |
| --- | --- | --- |
| `false_positive` | 7,500 | 75.0% |
| `true_positive` | 1,500 | 15.0% |
| `escalated` | 1,000 | 10.0% |

The 75% false-positive rate reflects real MDR operations where the majority of alerts from detection tooling are noise. True positives are a minority; escalated cases represent ambiguous situations requiring senior analyst judgment.

This balance is enforced as a target during generation. Each alert is scored with a continuous true-positive likelihood, and stratified sampling is applied to reach the target distribution — avoiding naive random label assignment while preserving realistic feature correlations.

---

## 3. MITRE Tactic Distribution

The five most common tactics in the generated dataset:

| Tactic | Share |
| --- | --- |
| Reconnaissance | 18.6% |
| Discovery | 18.4% |
| Initial Access | 8.0% |
| Execution | 7.0% |
| Privilege Escalation | 6.3% |

Reconnaissance and Discovery together account for ~37% of all alerts. This reflects the real-world pattern where early-stage detection tooling fires frequently on low-confidence behavioral patterns — logins, scans, enumeration attempts — most of which are benign.

Late-stage tactics such as Exfiltration and Impact are rare (~3% each), consistent with how infrequently attacks reach full execution in monitored environments.

---

## 4. Severity Signal Analysis

### Vendor vs inherent severity disagreement

| Metric | Value |
| --- | --- |
| Alerts with `\|vendor_severity − inherent_severity\| ≥ 2` | 33.4% of dataset |
| Pearson correlation between inherent and vendor severity | r = 0.13 |

The low correlation (r = 0.13) between inherent and vendor severity confirms that these are largely independent signals. More than one third of alerts show a substantial gap between what the detection tool labeled as urgent and what the underlying attack context implies. This disagreement is one of the most operationally significant patterns in the dataset — and the reason the model's severity conflict analysis is a key evaluation section.

### Burst index correlation

| Metric | Value |
| --- | --- |
| Correlation: `burst_index` vs false-positive label | r = 0.38 |

The burst index has the strongest direct linear correlation with the false-positive label of any single feature. Bursty alert patterns are a reliable noise indicator, which motivates its inclusion as an engineered feature and as one of the most important signals in the SHAP analysis.

### Vendor severity 5 (highest) behavior

| Metric | Value |
| --- | --- |
| Rate of vendor severity = 5 alerts | 11.1% |
| True positive rate within vendor severity = 5 alerts | 70.0% |

When the vendor assigns its maximum severity, 70% of those alerts turn out to be true positives. This gives the model a meaningful signal from vendor severity — but the 30% false-positive rate within this group means vendor severity alone is insufficient for safe auto-close decisions. The model must combine it with inherent context.

---

## 5. Asset Criticality

Asset criticality ranges from **1 to 10** across the dataset, distributed by asset type:

| Asset type | Criticality range | Rationale |
| --- | --- | --- |
| `domain_controller` | 8–10 | Always high-value targets |
| `server` | 5–10 | Medium to high business impact |
| `cloud_vm` | 4–8 | Medium impact, varies by workload |
| `workstation` | 2–6 | Low to medium; high noise volume |
| `iot` | 1–5 | Mostly low; limited blast radius |

This distribution allows the model to learn that the same alert firing on a domain controller is categorically more concerning than on an IoT device, without requiring an explicit asset-tier feature.

---

## 6. Interaction Effects

Two explicit interaction effects are embedded in the label generation logic. These are domain hypotheses encoded as scoring coefficients. The model is then tested on whether it rediscovers them.

### Effect 1 — Severity × Criticality → True Positive

**Formula:** `severity_norm × criticality_norm`

**Rationale:** Late-stage tactics firing on high-criticality assets are more likely to represent real threats. High inherent severity on a critical system is a compounding risk signal, not just additive.

**Empirical check:**

| Subset | TP rate |
| --- | --- |
| High severity (3–4) + high criticality (7–10) | **58.5%** |
| Overall baseline | 15.0% |

The high-severity/high-criticality subset has a TP rate nearly 4× the baseline. This interaction is captured by the `contextual_risk` derived feature.

---

### Effect 2 — Burstiness × Low Criticality → False Positive

**Formula:** `burst_index × (1 − criticality_norm)`

**Rationale:** Bursty alert storms on low-criticality assets (workstations, IoT) are characteristic of noisy rules, not real attacks. High volume on low-value targets is a strong false-positive indicator.

**Empirical check:**

| Subset | FP rate |
| --- | --- |
| High burst + low criticality (1–3) | **96.9%** |
| Overall baseline | 75.0% |

Nearly all alerts in this subset are noise. This pattern is captured by the `low_crit_burst` derived feature and confirmed as the strongest rightward-pushing signal in the SHAP analysis.

---

## 7. Derived Features

Three features are engineered during dataset generation and included as columns in the CSV. They are available to the model without requiring recalculation at training time.

### `contextual_risk = inherent_severity × asset_criticality`

Multiplies intrinsic threat severity by asset business impact. This is a multiplicative interaction rather than an additive one — if the probability of a true positive grows super-multiplicatively with both severity and criticality, the product increases class separability more than either feature alone.

**Expected direction:** higher values correlate with fewer false positives.

---

### `burst_index = log1p(alert_count_1h) / log1p(alert_count_24h)`

Measures short-term burstiness relative to daily volume. The `log1p` transformation stabilizes variance and bounds the extremes of count data. The ratio is scale-stable and bounded in [0, 1] when `count_1h ≤ count_24h`.

**Expected direction:** higher values correlate with more false positives.

---

### `severity_gap = inherent_severity − vendor_severity`

A signed residual capturing severity disagreement. Large negative values (vendor scores much higher than inherent context) imply vendor inflation — a pattern associated with noisy detections. Large positive values (inherent context much more severe than vendor scoring) suggest the tool may be under-alerting on a genuine threat.

**Expected direction:** large negative gaps indicate likely false positives; large positive gaps reduce FP probability.

---

## 8. Validation Warnings

The dataset generator runs self-validation checks and reports the following:

| Warning | Detail |
| --- | --- |
| Vendor severity 5 is not rare enough | Current rate is 11.1%, target was ≤ 10%. Vendor severity = 5 is slightly over-represented relative to real-world distributions where maximum severity is reserved for the rarest events. |
| Fewer than 8 rules have FP rates below 10% | The low-FP rule bucket produced fewer rules than the target count. This means the model has slightly less exposure to high-confidence, low-noise rules during training, which may contribute to modest false-positive suppression risk on those rule types. |

These warnings do not invalidate the dataset. They are surfaced as known imperfections that a reviewer or interviewer can ask about. In production, the generator configuration would be tuned to hit tighter targets before a training run.

---

*Generated by `generate_dataset.py`. Re-run with `--seed 42` to reproduce this exact dataset.*
