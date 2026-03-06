# Task 3 — Risk State Estimator

This document explains the design, mathematics, and operational interpretation of the rolling risk estimator built for Task 3 of the TekStream MDR assessment. The estimator produces a continuous risk score per customer or asset that decays when nothing happens and spikes (or accumulates) as new alerts arrive.

---

## 1. Why Per-Alert Classification Is Not Enough

The Task 2 model answers a narrow question: *is this individual alert a false positive?* That is necessary but not sufficient for operational risk management. Consider these two scenarios:

| Scenario | Individual alerts | Per-alert verdict | Actual risk |
| --- | --- | --- | --- |
| A | 1 high-severity alert on a domain controller | True positive | High — single critical event |
| B | 20 medium-severity alerts over 4 hours, progressively later kill-chain stages | Each borderline | Very high — coordinated campaign |
| C | 50 low-severity alerts in 10 minutes on IoT devices | Each likely FP | Low — bursty noisy rule, not an attack |

A per-alert classifier treats each row independently. The risk estimator tracks *cumulative state*: risk builds when alerts persist, spikes on severe events, and decays back to baseline when the environment is quiet. This gives analysts a prioritized customer or asset queue rather than an undifferentiated stream of per-alert verdicts.

---

## 2. Risk Score State

The estimator maintains a bounded continuous risk score:

```
R_t  ∈  [0, 1]
```

- **0** = no recent evidence of threat activity
- **1** = maximum risk; sustained or severe alert activity

---

## 3. Update Equation — Decay and Alert Contribution

Every time a new alert arrives for a customer or asset, the score updates in two steps.

### Step 1: Decay the existing score

```
R_decayed = R_prev × e^(−λ × Δt)
```

- `R_prev` is the score from the previous alert
- `Δt` is the elapsed time since that alert (in hours)
- `λ` is the decay rate, derived from a configurable half-life

This models the intuition that *threat evidence gets stale*. If nothing happens for 24 hours, the score should drop materially.

### Step 2: Incorporate the new alert

```
R_t = 1 − (1 − R_decayed) × (1 − Δ)
```

- `Δ` is the alert contribution (see Section 4)
- This formula keeps the score strictly in [0, 1] and makes each new alert push the score upward — but with diminishing marginal impact as the score approaches 1

The bounded update prevents runaway accumulation: once risk is near 1, additional alerts cannot push it above 1, while decay pulls it back whenever the environment quiets.

### Decay rate and half-life

```
λ = ln(2) / half_life_hours
```

The default half-life is 24 hours. This means a score of 0.80 decays to 0.40 after one quiet day, and to 0.20 after two days. The half-life is configurable to match the customer's environment — shorter for high-volatility environments where yesterday's alerts are irrelevant, longer for slow-burning APT scenarios.

---

## 4. Alert Contribution (Δ) — Severity-Weighted Signal

The alert contribution `Δ` is a weighted linear combination of normalized inputs, then clamped to [0, 0.95]:

```
Δ = w_inh × s_inh
  + w_vendor × s_vendor
  + w_context × c
  + w_behavior × b
  + w_disp × d

Δ = clamp(Δ, 0, 0.95)
```

### Input definitions

| Symbol | Feature | Normalization |
| --- | --- | --- |
| `s_inh` | Inherent severity | Normalized to [0, 1] from the 1–4 scale |
| `s_vendor` | Vendor severity | Normalized to [0, 1] |
| `c` | Asset criticality / context | Normalized to [0, 1] |
| `b` | Behavioral signal | Composite of burstiness + IOC match |
| `d` | Disposition confidence | 1.0 (true positive) → 0.5 (escalated) → 0.0 (false positive) |

### Default weights

| Component | Weight | Rationale |
| --- | --- | --- |
| Inherent severity | **0.45** | Primary signal — intrinsic threat severity dominates |
| Asset criticality | **0.20** | Business impact amplifies risk |
| Behavioral context | **0.15** | Burstiness + IOC presence confirm or dilute threat |
| Disposition confidence | **0.10** | Model verdict feeds back into risk state |
| Vendor severity | **0.10** | Intentionally weak — vendor scoring is noisy (r = 0.13) |

The weights encode a deliberate design choice: **vendor severity is not trusted as a primary signal**. The low correlation between vendor and inherent severity (r = 0.13, confirmed in the dataset validation) means vendor-assigned urgency is an unreliable anchor. Inherent severity and contextual evidence dominate.

The cap at `Δ = 0.95` (rather than 1.0) ensures that even a maximally severe single alert cannot push the risk score to 1.0 on its own — it takes sustained activity or a kill-chain progression to saturate the estimator.

---

## 5. Multi-Scale Risk Behavior

The formulation captures two operationally distinct patterns:

**Acute spike** — A single high-severity alert on a domain controller produces a large `Δ` and immediately elevates `R_t`. If no follow-on activity occurs, the score decays within 24–48 hours.

**Chronic accumulation** — Repeated medium-severity alerts (each with moderate `Δ`) stack because the decay between them is smaller than each new contribution. The score drifts upward over hours or days, eventually reaching a level that flags the customer for review even though no single alert was alarming.

This dual behavior is intentional: the model is sensitive to both *depth* (severity) and *breadth* (persistence) of threat signals.

---

## 6. Kill-Chain Progression Bonus

Beyond raw severity, a coordinated attack that advances through MITRE ATT&CK stages is inherently more dangerous than scattered activity at one stage. The estimator detects this using a kill-chain progression bonus.

### Stage mapping

Tactics are mapped to ordered stages:

| Stage | Tactics |
| --- | --- |
| 0 | Reconnaissance, Discovery |
| 1 | Initial Access |
| 2 | Execution, Persistence |
| 3 | Credential Access, Privilege Escalation |
| 4 | Lateral Movement, Command and Control |
| 5 | Exfiltration, Impact |

### Bonus computation

Within a configurable recent window (default 24 hours), the estimator counts ordered stage pairs — transitions from a lower stage to a higher stage:

```
bonus = γ × (progression_pairs / max_pairs)
Δ = clamp(Δ × (1 + bonus), 0, 0.95)
```

- `γ = 0.4` (default scaling coefficient)
- `max_pairs` is the maximum possible ordered pairs from the observed stages
- The bonus multiplies `Δ`, so it amplifies an already-significant alert contribution rather than adding a fixed amount

**Why pairs, not a full sequence?** A full sequence model would require a complete kill-chain observation, which rarely happens cleanly in real environments. Counting ordered pairs is a lightweight heuristic that rewards forward progression without requiring an unbroken chain. A customer that shows Recon → Initial Access → Privilege Escalation within 24 hours gets a meaningful bonus even if Execution alerts were missed or absent.

---

## 7. Output Visualizations

### Risk Timeline

![Risk Timeline](outputs/risk_timeline.png)

The plot shows `R_t` over time (x-axis: date, y-axis: risk score 0–1) for the busiest customer in the dataset, from December 2025 to March 2026.

Key observations:
- **The black line** is the EWMA risk state. It begins near zero for the first alert, then rapidly climbs to the 0.9–1.0 range as alerts accumulate and stays elevated for the full period.
- **Salmon/orange dots** represent individual alerts colored by disposition (false positive in this customer's data).
- **"Elevated risk period: progression into later kill-chain stages"** — the annotation marks a window where alerts advance through multiple MITRE stages, triggering the kill-chain bonus and sustaining a high risk state.
- **"Kill-chain progression bonus"** — labels the exact point where the bonus fires, showing the score holds at its peak rather than beginning to decay.
- **"False alarm period: bursty low-criticality FP cluster"** — marks a burst of false positives at the end of the timeline. The risk score does not spike here because these alerts have low criticality and high burstiness, which suppresses `Δ`. The estimator correctly does not treat a noise storm as elevated risk.

The plot visualizes the core behavioral contract: the model rises on genuine threat progression and resists false inflation from bursty noise.

---

### Decayed Risk vs. Alert Contribution

![Decayed Risk vs Alert Contribution](outputs/risk_decay_vs_delta.png)

This scatter plot shows the relationship between `R_decayed` (x-axis) and `Δ` (y-axis) for every alert update.

Key observations:
- **The two isolated points at low decayed risk** (x ≈ 0.0 and x ≈ 0.2) represent the first two alerts in the customer's history, where no prior risk state exists to decay from.
- **The dense cluster at high decayed risk (0.8–1.0)** is the steady state. Once the risk score is elevated, most subsequent updates start from a high decayed value — meaning the score never fully recovers to baseline between alerts.
- **The vertical spread in Δ (0.3–0.95)** at high decayed risk shows that individual alerts still contribute meaningfully different amounts depending on their severity and context. High-Δ alerts (top right) are severe events arriving into an already-elevated state — the highest-priority moments for analyst review.
- **The bounded update formula** is visible in the distribution: points in the top-right corner (high decayed risk + high Δ) still produce R_t values near but not exceeding 1.0, confirming the ceiling behavior.

---

## 8. Operational Interpretation

| Signal | What it means for the analyst |
| --- | --- |
| Rising `R_t` over hours | Ongoing alert activity — customer needs active monitoring |
| Sustained `R_t` near 1.0 | Persistent threat environment — consider escalation |
| `R_t` decaying smoothly | Quiet period — safe to deprioritize |
| High `R_t` + kill-chain annotation | Multi-stage progression — potential active campaign |
| High `R_t` despite known FP burst | Check if burst suppressed Δ correctly; model should resist noise |

The risk estimator is designed to be **explainable at each step**: every update decomposes into the four contributing signals (severity, context, behavior, disposition) plus the optional kill-chain bonus. An analyst can always trace why `R_t` is at its current value by examining the last N updates.

**Visualization default:** the timeline plot defaults to the busiest customer so the most active risk trajectory is immediately visible during demos or code review.

---

## 9. Configuration and Extension

| Parameter | Default | Effect |
| --- | --- | --- |
| `half_life_hours` | 24 | Controls how fast risk decays in quiet periods |
| `gamma` (kill-chain bonus) | 0.4 | Controls sensitivity to kill-chain progression |
| `progression_window_hours` | 24 | Lookback window for stage pair detection |
| Δ cap | 0.95 | Prevents single-alert saturation |
| Weight vector | [0.45, 0.10, 0.20, 0.15, 0.10] | Tunable per customer environment |

In production, these parameters would be customer-specific, tuned by observed false-positive rates and analyst feedback during the first weeks of deployment.
