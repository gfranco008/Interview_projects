# TekStream AI/ML Architect Technical Assessment

This folder contains the assessment work product for TekStream Solutions' AI/ML Architect technical assessment focused on a cybersecurity intelligence platform.

## Summary
You will build a synthetic security alert dataset, train a calibrated false positive predictor, and (optionally) design a dynamic risk state estimator that tracks risk over time. The work emphasizes realistic feature correlations, explainability, calibrated confidence, and externally configurable thresholds.

## Tasks
1. Data Generation and Feature Engineering (Required)
- Generate a 10,000-row synthetic alert dataset with realistic distributions and correlations.
- Implement at least two interaction effects and document them.
- Provide a brief statistical summary.
- Engineer three derived features that capture cross-axis interactions.

2. False Positive Predictor (Required)
- Train and compare at least two model approaches.
- Calibrate probabilities and show a reliability diagram.
- Externalize decision thresholds in a config file.
- Produce a thorough evaluation report including severity conflict analysis.

3. Dynamic Risk State Estimation (Stretch)
- Define a mathematical formulation for rolling risk estimation.
- Implement a risk estimator that updates over time.
- Visualize a risk timeline and annotate key periods.

## Expected Deliverable Structure
```
assessment/
  README.md
  requirements.txt
  policy.yaml
  generate_dataset.py
  fp_predictor.py
  evaluation.md
  risk_model.md
  risk_estimator.py
  outputs/
    alerts_dataset.csv
    calibration_curve.png
    roc_curves.png
    feature_importance.png
    threshold_analysis.png
    risk_timeline.png
```

## Constraints and Guidance
- Python is the required language. Choose libraries and justify them.
- Thresholds must be loaded from external configuration.
- Every prediction must be explainable and probabilities must be calibrated.
- Severity conflict analysis is required: assess cases where vendor and inherent severity strongly disagree.

## Proposed Solution Ideas
Task 1 (Synthetic Data + Features)
- Generate tactics with a skewed categorical distribution (Discovery and Reconnaissance frequent; Impact and Exfiltration rare).
- Assign 3-5 techniques per tactic with technique-specific FP rates.
- Model correlations explicitly: asset_type drives asset_criticality; bursty alert_count_1h and short time_since_last_alert increase FP likelihood; IOC matches correlate with TP.
- Include interaction effects like:
  - high inherent_severity combined with high asset_criticality raises TP probability even when vendor_severity is low.
  - high alert burstiness on low-criticality assets increases FP likelihood.
- Derived features (examples):
  - risk_alignment = normalized(inherent_severity - vendor_severity)
  - contextual_risk = inherent_severity * asset_criticality
  - burst_index = log1p(alert_count_1h) / log1p(alert_count_24h)

Task 2 (FP Predictor)
- Compare a linear model (logistic regression) against a tree-based model (gradient boosting or random forest) to capture interactions.
- Use stratified train/validation/test splits to preserve class balance.
- Calibrate with isotonic regression or Platt scaling; plot pre/post calibration curves.
- Externalize thresholds in YAML and generate a threshold sweep report for workload and error tradeoffs.
- Explainability: feature importance (permutation or SHAP) and severity conflict subset analysis.

Task 3 (Risk State Estimation)
- Use an EWMA or Bayesian update model where alerts contribute a weighted risk increment based on inherent severity and kill-chain progression.
- Implement time decay to reduce risk in quiet periods.
- Add a progression bonus when a sequence advances from early to late tactics within a time window.
- Visualize a 90-day risk timeline with annotated bursts and false alarms.
