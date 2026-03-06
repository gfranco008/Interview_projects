# Task 3 Risk Model

## 1. Objective
Per-alert classification is not enough for operational risk. We need a rolling state that decays in quiet periods, spikes on severe alerts, and accumulates when medium alerts persist.

## 2. State Definition
We maintain a continuous risk score:

```
R_t ∈ [0, 1]
```

## 3. Update Equation (Decay + Alert Contribution)
Let `Δt` be the time since the previous alert and `λ` be the decay rate.

```
R_decayed = R_prev * e^(-λΔt)
R_t = 1 - (1 - R_decayed) * (1 - Δ)
```

This keeps the score bounded in [0, 1] while allowing additive influence from new alerts.

Decay rate uses half-life:

```
λ = ln(2) / half_life_hours
```

## 4. Alert Contribution Δ (Severity-Weighted)
Alert contribution is a weighted sum of normalized inputs, then clamped:

```
Δ = w_inh * s_inh
  + w_vendor * s_vendor
  + w_context * c
  + w_behavior * b
  + w_disp * d

Δ = clamp(Δ, 0, 0.95)
```

Where:
- `s_inh`: inherent severity normalized to [0,1] using the 1–4 scale
- `s_vendor`: vendor severity normalized to [0,1]
- `c`: asset criticality normalized to [0,1]
- `b`: behavioral signal (burstiness + IOC match)
- `d`: disposition confidence (true_positive > escalated > false_positive)

Default weights:

- inherent severity: 0.45
- vendor severity: 0.10
- asset criticality/context: 0.20
- behavioral context: 0.15
- disposition confidence: 0.10

**Bias control:** vendor severity is intentionally weak. This ensures inherent severity and contextual evidence dominate.

## 5. Multi-Scale Risk
The formulation supports both:

- **Single critical alert:** high inherent severity on a critical asset produces a large Δ and immediate spike.
- **Accumulation:** repeated medium alerts stack because decay is gradual.

This yields both acute spikes and chronic build-up.

## 6. Kill Chain Progression Bonus
We add a bonus when alerts progress through the kill chain within a recent window (default 24h).

Define an ordered stage mapping:

```
Recon/Discovery → Initial Access → Execution/Persistence
→ Credential Access/Priv Esc → Lateral Movement/C2 → Exfiltration/Impact
```

We compute a progression ratio from ordered stage pairs in the window:

```
bonus = γ * (progression_pairs / max_pairs)
Δ = clamp(Δ * (1 + bonus), 0, 0.95)
```

This rewards forward progression without requiring a full sequence model.
It is a lightweight progression heuristic based on ordered unique stage advancement within a recent window.

## 7. Operational Interpretation
- **Analyst prioritization:** rising `R_t` flags customers or assets requiring review.
- **Health trend:** sustained risk suggests ongoing activity.
- **Explainability:** each update is decomposable into severity, context, behavior, and progression.
- **Visualization default:** the plot defaults to the busiest customer so the trajectory is easy to inspect.

## Outputs
- `assessment/outputs/risk_timeline.png` showing `R_t` over time with annotations for elevated-risk and false-alarm periods.
- `assessment/outputs/risk_decay_vs_delta.png` showing decayed risk vs. alert contribution (Δ).
