# TekStream Assessment - Task 1

This folder contains the synthetic data generator for Task 1 of the TekStream AI/ML Architect assessment. It follows the PDF guidance and the additional LLM notes you provided.

How to run

```
python generate_dataset.py --rows 10000 --seed 42
```

Optional extras

- Add SOC-style context columns with `--extra-columns`.
- Override defaults with `--config path/to/config.json`.

Validation

```
python validate_dataset.py --rows 2000
```

Add `--strict` to exit non-zero if any validation warnings are found.

Outputs

- `assessment/outputs/alerts_dataset.csv`
- `assessment/outputs/summary.txt`

Interaction effects implemented

- High inherent severity combined with high asset criticality increases true positive likelihood. This models late-stage tactics on high-value assets being inherently risky.
- High burstiness combined with low asset criticality increases false positive likelihood. This models noisy alert storms on low-value assets.

Derived features (included as columns)

- `contextual_risk = inherent_severity * asset_criticality`.
Justification: multiplies threat intrinsic severity by business impact; monotone in both factors.
- `burst_index = log1p(alert_count_1h) / log1p(alert_count_24h)`.
Justification: bounded, scale-stable burstiness ratio.
- `severity_gap = inherent_severity - vendor_severity`.
Justification: signed residual capturing severity disagreement (negative gaps imply vendor inflation).
- `low_crit_burst = burst_index * (1 - asset_criticality / 10)`.
Justification: interaction that emphasizes burstiness on low-criticality assets.
- `ioc_weighted_risk = has_ioc_match * inherent_severity * asset_criticality`.
Justification: IOC gate on risk that scales with inherent severity and asset impact.

Key distribution assumptions

- MITRE tactic distribution is skewed toward Reconnaissance and Discovery; Impact and Exfiltration are rare.
- 50 `source_rule` values are created with a mix of high-FP and low-FP rules (>80% and <10% base FP rates).
- 3 to 5 MITRE techniques per tactic with technique-level FP rates aligned to tactic severity.
- Asset criticality is correlated with asset type; domain controllers are rare and highly critical.
- Vendor severity is centered on 3, with severity 5 rare; it intentionally disagrees with inherent severity to reflect real-world inconsistency.
- False positives are biased toward business hours; true positives are uniform across the day.
- IOC matches are generated at ~40% for true positives and ~3% for false positives (escalated is intermediate).


