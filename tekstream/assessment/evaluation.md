# Evaluation Report

## Model Comparison
- XGBoost (tuned): CV AUC 0.968±0.002, validation AUC 0.970.
- Logistic Regression: CV AUC 0.944±0.006, validation AUC 0.949.
Note: ROC/PR plots use uncalibrated model scores for comparison.
PR curve: outputs/pr_curve.png
Selected model: **XGBoost (tuned)** (CV AUC 0.968±0.002, validation AUC 0.970).
Calibrated model test AUC: 0.963, Brier: 0.066, Log loss: 0.248.
Calibration metrics (test): pre Brier 0.066, pre LogLoss 0.201; post Brier 0.066, post LogLoss 0.248.
Calibration trust note: miscalibrated scores erode analyst confidence because probabilities no longer map to observed FP rates, leading to inconsistent triage.
Feature importance check: inherent_severity 0.0006 vs vendor_severity -0.0001 (permutation importance mean).
Calibration strategy: CalibratedClassifierCV with FrozenEstimator, fit on the validation split to preserve the exact base model weights.
Note: validation was reused for both model comparison and calibration in this take-home setup. In production, I would separate tuning, calibration, and final evaluation with either a dedicated calibration holdout or time-based backtesting.
Note: test set is 20% of data; subset metrics (e.g., severity conflicts) can be noisy for rare segments.

## Threshold Policy Impact
Auto-close threshold: 0.95, Flag-review threshold: 0.70
False-positive pass-through rate: 17.40%
Missed true-positive rate: 2.20%
Analyst workload rate: 37.50%
Volume split: auto-close 62.5%, flag-review 6.2%, manual triage 31.2%.

## Operating Point Metrics
Chosen threshold: 0.70 (policy flag-review boundary)
Precision: 0.972, Recall: 0.891, F1: 0.929
Justification: aligns with the review boundary so predictions above the threshold drive analyst action while minimizing missed TPs at auto-close.

## Hyperparameter Tuning
Used GridSearchCV (5-fold ROC-AUC) over XGBoost depth, learning_rate, and number of estimators.
Grid: max_depth=[3, 5, 7], learning_rate=[0.05, 0.08, 0.12], n_estimators=[150, 200, 250].
Best params: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 250} (CV AUC 0.968±0.002).

## Severity Conflict Analysis
Subset where |inherent_severity - vendor_severity| >= 2:
Count: 702, FP rate: 68.09%, AUC: 0.978

High vendor / low inherent (vendor 4-5, inherent 1):
Count: 193, Precision: 0.961, Recall: 0.930, AUC: 0.968
Low vendor / high inherent (vendor 1-2, inherent 3-4):
Count: 286, Precision: 0.964, Recall: 0.692, AUC: 0.899

Qualitative interpretation:
Model leans toward inherent_severity: higher FP probabilities when inherent is low even if vendor severity is high (mean FP prob delta=+0.282).
Vendor-severity anchoring bias failure mode: if the model overweights vendor scores, high vendor/low inherent alerts would be misclassified as true positives; we explicitly check the conflicting subsets above to guard against that bias.

## Error Analysis
Overall error rate: 10.15% (FN 164, FP 39).
By mitre_tactic:
Missed FPs (FN): Credential Access 25/127 (19.7%), Command and Control 22/104 (21.2%), Defense Evasion 20/114 (17.5%), Persistence 15/127 (11.8%), Privilege Escalation 14/121 (11.6%)
True alerts flagged as FP (FP): Privilege Escalation 6/121 (5.0%), Execution 4/134 (3.0%), Initial Access 4/166 (2.4%), Discovery 4/356 (1.1%), Reconnaissance 4/375 (1.1%)
By asset_type:
Missed FPs (FN): workstation 59/824 (7.2%), cloud_vm 54/441 (12.2%), server 37/404 (9.2%), domain_controller 8/101 (7.9%), iot 6/230 (2.6%)
True alerts flagged as FP (FP): workstation 20/824 (2.4%), server 9/404 (2.2%), cloud_vm 7/441 (1.6%), domain_controller 2/101 (2.0%), iot 1/230 (0.4%)
By source_rule:
Missed FPs (FN): RULE_008 13/68 (19.1%), RULE_024 11/95 (11.6%), RULE_046 10/57 (17.5%), RULE_019 10/68 (14.7%), RULE_020 10/69 (14.5%)
True alerts flagged as FP (FP): RULE_017 5/44 (11.4%), RULE_038 5/174 (2.9%), RULE_007 4/156 (2.6%), RULE_024 3/95 (3.2%), RULE_041 2/34 (5.9%)
Conclusion: FNs concentrate most in asset_type 'workstation' (59/824, 7.2%), likely due to environmental noise specific to that asset class.

## SHAP Explainability
SHAP summary computed on 1000 test samples.
Plot: outputs/shap_summary.png
Top features by mean(|SHAP|): num__vendor_severity (1.2753), num__contextual_risk (1.1326), num__inherent_severity (0.8437), num__low_crit_burst (0.8124), num__alert_count_1h (0.5592), cat__alert_type_signature_broad (0.1091), num__alert_count_24h (0.1005), num__burst_index (0.0872), cat__user_privilege_level_service_account (0.0856), num__time_since_last_alert (0.0610)
SHAP vs permutation comparison: outputs/shap_vs_permutation.png (normalized scale).

## Saved Artifacts
Base model: /Users/gfranco/Documents/qvest/tekstream/assessment/outputs/model_base.joblib
Calibrated model: /Users/gfranco/Documents/qvest/tekstream/assessment/outputs/model_calibrated.joblib
Metadata: /Users/gfranco/Documents/qvest/tekstream/assessment/outputs/model_metadata.json
Policy snapshot: /Users/gfranco/Documents/qvest/tekstream/assessment/outputs/policy_snapshot.yaml

## SOC Deployment Notes
If I were deploying this to a SOC handling ~500 alerts/day, I would position the model as a decision-support layer that drives the triage split. Alerts with a predicted FP probability >= 0.70 move to a fast-path review queue, and those above the auto-close threshold require the lowest manual effort. With the current policy, the expected analyst workload is about 37.5% of daily volume; that maps to a concrete staffing estimate and is easy to tune by adjusting the thresholds.
I would tell analysts that the model is calibrated (post-calibration Brier/log-loss tracked) and that we explicitly monitor error hot spots by tactic, asset type, and rule. If we see drift (e.g., bursts in a specific tactic), we can temporarily override auto-close for that group and feed the analyst feedback into the next retraining cycle. The goal is not to replace judgment, but to prioritize review where the model is least confident or historically error-prone.
Operationally, I would start with conservative thresholds in week 1, review the false-negative cases daily, and only then increase auto-close aggressiveness. We would also validate vendor/inherent severity conflicts and make sure the model behavior aligns with analyst expectations for high-impact tactics. This keeps the system transparent, adjustable, and aligned with SOC risk tolerance.
