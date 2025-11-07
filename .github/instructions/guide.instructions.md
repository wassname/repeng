---
applyTo: '**'
---

## Data Construction Example

Using last 6 tokens of sequences with minimal differences:
```python
cho = "I love cheese; let me tell you about the andes mountains"
rej = "I hate cheese; let me tell you about the andes mountains"
```

# Steering Vector Extraction Guide

See README.md for overview of key ideas (reasoning trajectory hypothesis, last-token extraction, gradient-to-steering mapping, layer-specific steering).

Jargon
- hs = W @ x + b = hidden states (pre-activation) at a given layer L (could be residual space, up_proj, down_proj, k, v, etc)
- hs_pi_cho: activations from layer L for the policy model on the chosen prompt
- hs_ref_rej: likewise for the reference model on the rejected prompt
- V_w, S_w, U_w = SVD(L.W) - singular vectors of weight matrix for layer L
- dHS = hs_pi_cho - hs_ref_rej
- V_dhs, S_dhs, U_dhs = SVD(dHS)
- dHSU = dHS @ U_dhs



**Why this works**: Sequences differ by one word early on but must have different hidden states at the end if the model plans to continue them differently. The model's internal state at position T must encode its plan for positions T+1, T+2, ... This captures the planning signal for steering.

Extract from last non-padded token because it represents the model's "state of mind" for continuing the trajectory. Contrasting minimally different sequences here amplifies conceptual differences (honesty vs dishonesty) while controlling surface features.

## Loss Function Details

Loss in `repeng/train/inner_contrastive_loss.py but it seeks to seperate the activations for a layer along a preference direction. But it is bounded by a coherence constraint to ensure generation quality. The logp_pi should be better than logp_ref for each token by a margin, this defined a region of coherence. This must be per token to stop the model gaming this (e.g. making " the" very likely and downing our coherence degredation in all other tokens)

See `docs/loss_geometry.md` for detailed geometric explanation with diagram (may be out of date).


# psudo code for extracting steering vectors

```py
# DATA: Contrastive pairs differing in 1-6 tokens
honest    = ["I love cheese; let me tell you about the andes mountains", ...]
dishonest = ["I hate cheese; let me tell you about the andes mountains", ...]
batch = [honest[0], dishonest[0], honest[1], dishonest[1], ...]

# SETUP: Low-rank SVD decomposition with learnable rotations + scaling
for layer in model.target_layers:
    U, Σ, V = SVD(layer.W)[:r]
    W_res = W - U @ Σ @ V.T
    θ_v = init_skew_symmetric(r)
    λ = rand(r) # must init non-zero to break symmetry

def forward(x, layer, c):  # c ∈ {-1, +1} steers honest ↔ dishonest
    R = cayley(θ_v, c)
    Σ_c = exp(c · λ) ⊙ Σ
    return x @ (V @ R) @ Σ_c @ U.T + x @ W_res.T

# TRAINING: Contrastive loss for reversible steering
for batch in dataloader:
    h_ref = model(batch, c=0)
    l_total = 0
    
    for c in [-1, +1]:
        h = model(batch, c=c)
        h_pos, h_neg = h[::2], h[1::2]
        
        Δ = (h_pos - h_neg).mean() @ d_steer  # Maximize separation
        l_total += -c · Δ + λ_coh · |logp(h) - logp(h_ref)|  # + coherence
    
    l_total.backward()
    update(θ_v, λ)
```
