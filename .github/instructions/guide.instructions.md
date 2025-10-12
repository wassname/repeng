---
applyTo: '**'
---

# Steering Vector Extraction Guide

See README.md for overview of key ideas (reasoning trajectory hypothesis, last-token extraction, gradient-to-steering mapping, layer-specific steering).

This guide covers implementation-specific details and empirical results.

## Data Construction Example

Using last 6 tokens of sequences with minimal differences:
```python
pos = "I love cheese; let me tell you about the andes mountains"
neg = "I hate cheese; let me tell you about the andes mountains"
```

**Why this works**: Sequences differ by one word early on but must have different hidden states at the end if the model plans to continue them differently. The model's internal state at position T must encode its plan for positions T+1, T+2, ... This captures the planning signal for steering.

Extract from last non-padded token because it represents the model's "state of mind" for continuing the trajectory. Contrasting minimally different sequences here amplifies conceptual differences (honesty vs dishonesty) while controlling surface features.

## Loss Function Details

Loss in `repeng/train/inner_contrastive_loss.py::contrastive_steering_loss_noref`:

```python
pref_dir = hs_pos - hs_neg
pref_mag = torch.norm(pref_dir, dim=-1)
baseline_logp = logp_avg_pos_label.detach()
coherence_penalty = F.relu((baseline_logp * 0.99 - logp_avg_pos_label))**2 * 10000
loss = -pref_mag.mean() + coherence_penalty.mean()
```

**Design rationale**: Direction is frozen (computed on base model or adapter with coeff=0), so adapter can't game the metric by rotating direction. It must genuinely separate concepts. The 0.99 factor positions computation just inside coherence boundary so both terms contribute gradients. Heavy weighting (10000x) balances with high-magnitude separation term.

**Per-token application**: We apply the margin per token, not sequence-level logprobs. This prevents the model from gaming the metric by focusing on a few tokens.

**Geometric intuition**: The loss defines a coherence boundary (quadratic penalty outside it) and maximizes separation along a frozen preference direction. Direction extracted via PCA/Fisher from reference model (unlearnable). Gradients point in opposite directions for pos/neg, creating a clear preference axis while maintaining generation quality.

See `docs/loss_geometry.md` for detailed geometric explanation with diagram.

## Layer Selection Results

Performance varies dramatically by layer type (score = composite of slope, R², validity):

**Best performers (Qwen-4B-Thinking, hidden states)**:
- `fisher_steer_cov_reg1`: **score 15.93** (slope -18.01, R² 0.88)
- `fisher_steer_dual`: score 8.86 (slope -21, R² 0.66)
- `fisher_steer_reg2`: score 6.75 (slope -19.36, R² 0.54)

**Attention sublayers** (worse than hidden states):
- `k_proj`: score 1.42 (extrapolates well but disconnected from hs)
- `v_proj`: score 0.59 (noisy, poor coupling)
- `o_proj`: Writes to residual, should be better (untested)
- `q_proj`: Changes attention focus, can be strong but noisy (untested)

**MLP sublayers** (mixed, some very strong):
- `up_proj`: score 13.99 (slope 17.92, R² 0.78)
- `down_proj`: score 8.54 (writes to residual, monotone dose-response)
- `gate_proj`: score 5.66 (nonlinear/saturated)

**Architecture guidance**: For controllable behavior, prefer layers that write directly to residual stream (o_proj, down_proj) over those affecting attention patterns (k/v). Results confirm this: k/v are weaker than hidden states, while up/down_proj are strong.

## Method Comparison

Best methods use Fisher natural gradients with covariance FIM and regularization:
- `fisher_steer_cov_reg1`: Empirical FIM from gradient outer products, reg 1e-5
- `fisher_steer_dual`: Dual pos/neg directions for balanced steering
- PCA (baseline): Simple difference mean, often competitive but less robust

Regularization matters: reg1 (1e-5) >> reg2 (1e-4) >> reg3 (1e-3). Too much damping loses curvature info.
