"""
Weight SVD-based steering: directly manipulate singular values of layer weights.

For a layer with weight matrix W, we decompose:
    W = U @ Σ @ V.T

Then steering is done by:
    Layer(x) + x @ U @ (ΔΣ * coeff) @ V.T

where ΔΣ is learned from activations.

## Integration Plan

Current architecture challenge: `read_representations()` only receives activations/gradients,
not the model itself. Weight SVD requires access to layer.weight matrices.

### Option 1: Pass model to read_representations
```python
def read_representations(act, grads, model, ...):
    for layer in layers:
        weight = model.get_submodule(layer).weight
        delta_sigma = weight_svd_steering(weight, act_pos, act_neg)
```
Pros: Clean, direct
Cons: Changes API, couples to model structure

### Option 2: Pre-extract weights in ControlVector.train()
```python
def train(model, ...):
    weights = {layer: model.get_submodule(layer).weight for layer in layers}
    dirs = read_representations(act, grads, weights=weights, ...)
```
Pros: Minimal API change (optional kwarg)
Cons: Extra memory, weights dict separate from act/grads

### Option 3: Hook-based application (defer to control.py)
Store ΔΣ as direction, apply during inference via custom hook that:
1. Computes W = U @ Σ @ V.T on first call (cache)
2. Applies x @ U @ diag(ΔΣ * coeff) @ V.T as steering

Pros: Separates extraction from application
Cons: More complex control.py, SVD at inference time

### Recommendation
Start with Option 2 for extraction, then implement Option 3 for efficient application.
This keeps ControlVector.train() changes minimal while enabling proper weight-space steering.

TODO: Implement weight extraction and application hooks
"""
import torch
from torch.linalg import svd
from jaxtyping import Float
from torch import Tensor
from typing import Literal


def weight_svd_steering(
    weight: Float[Tensor, "dim_out dim_in"],  # layer weight matrix
    activations_pos: Float[Tensor, "batch/2 dim_in"],
    activations_neg: Float[Tensor, "batch/2 dim_in"],
    mode: Literal["diff", "pos_only"] = "diff",
    normalize: bool = True,
) -> Float[Tensor, "rank"]:
    """
    Compute steering vector in singular value space.
    
    For PyTorch Linear layers: y = x @ W.T where W is [out_features, in_features]
    SVD: W = U @ Σ @ V.T where U is [out_features, rank], V is [in_features, rank]
    
    This function works with **output activations** y (what we capture from layers).
    We project y into U-space (output singular vectors) to get coordinates in singular space.
    
    Args:
        weight: Weight matrix of the layer [out_features, in_features]
        activations_pos: Output activations for positive examples [batch/2, out_features]
        activations_neg: Output activations for negative examples [batch/2, out_features]
        mode: How to compute steering direction
            - "diff": Use difference in U-space between pos/neg
            - "pos_only": Use only positive activations
        normalize: Whether to normalize ΔΣ to unit norm
    
    Returns:
        ΔΣ: Steering vector in singular value space [rank]
    """
    # SVD of weight matrix: W = U @ Σ @ V.T
    U, S, Vt = svd(weight, full_matrices=False)  # U: [out_features, rank], S: [rank], Vt: [rank, in_features]
    rank = S.shape[0]
    
    # Project output activations into U-space (output singular vectors)
    # y is in row-space of W (spanned by U), so y @ U.T gives coords in U-basis
    # But U is orthonormal so y @ U also works (U.T @ U = I)
    pos_in_U = activations_pos @ U  # [batch/2, rank]
    neg_in_U = activations_neg @ U  # [batch/2, rank]
    
    if mode == "diff":
        # Difference in U-space (what changes between pos/neg)
        delta_U = (pos_in_U - neg_in_U).mean(0)  # [rank]
    elif mode == "pos_only":
        # Just the positive direction
        delta_U = pos_in_U.mean(0)  # [rank]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # This becomes our ΔΣ: how much to scale each singular component
    delta_sigma = delta_U
    
    if normalize:
        norm = torch.norm(delta_sigma)
        if norm > 1e-8:
            delta_sigma = delta_sigma / norm
    
    return delta_sigma


def apply_weight_svd_steering(
    x: Float[Tensor, "... dim_in"],
    weight: Float[Tensor, "dim_out dim_in"],
    delta_sigma: Float[Tensor, "rank"],
    coeff: float = 1.0,
) -> Float[Tensor, "... dim_out"]:
    """
    Apply weight SVD steering to activations.
    
    Computes: x @ U @ diag(delta_sigma * coeff) @ V.T
    
    This is added to the normal layer output.
    """
    U, S, Vt = svd(weight, full_matrices=False)
    
    # x @ V: project to V-space
    x_V = x @ Vt.T  # [..., rank]
    
    # Scale by delta_sigma
    x_V_scaled = x_V * (delta_sigma * coeff)  # [..., rank]
    
    # Project back: @ U.T
    steering = x_V_scaled @ U.T  # [..., dim_out]
    
    return steering
