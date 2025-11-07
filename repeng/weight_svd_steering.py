"""
Weight SVD-based steering: directly manipulate singular values of layer weights.

For a layer with weight matrix W, we decompose:
    W = U @ Σ @ V.T

Then steering is done by:
    Layer(x) + x @ U @ (ΔΣ * coeff) @ V.T

where ΔΣ is learned from activations.
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
    
    The idea: project activations into U-V basis, compute difference in that space.
    Returns ΔΣ that can be used as: x @ U @ diag(ΔΣ * coeff) @ V.T
    
    Args:
        weight: Weight matrix of the layer (e.g., k_proj.weight, up_proj.weight)
        activations_pos: Activations for positive (preferred) examples
        activations_neg: Activations for negative (non-preferred) examples  
        mode: How to compute steering direction
            - "diff": Use difference in U-space between pos/neg
            - "pos_only": Use only positive activations
        normalize: Whether to normalize ΔΣ to unit norm
    
    Returns:
        ΔΣ: Steering vector in singular value space [rank]
    """
    # SVD of weight matrix
    U, S, Vt = svd(weight, full_matrices=False)  # U: [dim_out, rank], S: [rank], Vt: [rank, dim_in]
    rank = S.shape[0]
    
    # Project activations into V-space (input side)
    # h @ W = h @ V @ Σ @ U.T, so h @ V gives us coords in V-basis
    pos_in_V = activations_pos @ Vt.T  # [batch/2, rank]
    neg_in_V = activations_neg @ Vt.T  # [batch/2, rank]
    
    if mode == "diff":
        # Difference in V-space (what changes between pos/neg)
        delta_V = (pos_in_V - neg_in_V).mean(0)  # [rank]
    elif mode == "pos_only":
        # Just the positive direction
        delta_V = pos_in_V.mean(0)  # [rank]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # This becomes our ΔΣ: how much to scale each singular component
    delta_sigma = delta_V
    
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
