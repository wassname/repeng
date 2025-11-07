"""
KFAC-style steering using separate input/output covariances.

Based on Kronecker-Factored Approximate Curvature (KFAC):
https://arxiv.org/html/2510.24256v2

For layer with h_out = h_in @ W:
- A = Cov(h_in): input-side covariance 
- G = Cov(∂loss/∂h_out): output-side (gradient) covariance
- Fisher ≈ G ⊗ A (Kronecker product approximation)

For steering, we decompose each and use their eigenvectors as basis.
"""
import torch
from torch.linalg import svd
from jaxtyping import Float
from torch import Tensor
from typing import Literal


def kfac_steering(
    activations: Float[Tensor, "batch dim_in"],  # h_in for this layer
    gradients: Float[Tensor, "batch dim_out"],  # ∂loss/∂h_out (backprop gradient)
    lambda_reg: float = 1e-2,
    mode: Literal["input", "output", "product"] = "product",
    rank: int | None = None,  # how many eigenvectors to use
) -> Float[Tensor, "dim_in"] | Float[Tensor, "dim_out"]:
    """
    Compute KFAC-based steering direction.
    
    Args:
        activations: Input activations h_in to the layer
        gradients: Gradient of loss w.r.t. layer output ∂loss/∂h_out
        lambda_reg: Regularization for covariance inversion
        mode: Which covariance to use for steering
            - "input": Use input covariance A (returns dim_in vector)
            - "output": Use gradient covariance G (returns dim_out vector)  
            - "product": Use their product A @ G (experimental, returns dim_in)
        rank: Number of top eigenvectors to keep (None = full rank)
    
    Returns:
        Steering direction vector
    """
    device = activations.device
    batch_size = activations.shape[0]
    
    # Compute input covariance A = Cov(h_in)
    A = torch.einsum('bi,bj->ij', activations, activations) / batch_size
    A = A + lambda_reg * torch.eye(A.shape[0], device=device)
    
    # Compute gradient covariance G = Cov(∂loss/∂h_out)
    G = torch.einsum('bi,bj->ij', gradients, gradients) / batch_size
    G = G + lambda_reg * torch.eye(G.shape[0], device=device)
    
    if mode == "input":
        # Use input-side eigenvectors
        U_a, S_a, _ = svd(A)
        if rank:
            U_a = U_a[:, :rank]
            S_a = S_a[:rank]
        # Weight by eigenvalues (directions of high variance)
        direction = U_a @ torch.diag(S_a) @ U_a.T @ activations.mean(0)
        
    elif mode == "output":
        # Use gradient-side eigenvectors
        U_g, S_g, _ = svd(G)
        if rank:
            U_g = U_g[:, :rank]
            S_g = S_g[:rank]
        # Weight by eigenvalues (directions of high curvature)
        direction = U_g @ torch.diag(S_g) @ U_g.T @ gradients.mean(0)
        
    elif mode == "product":
        # Experimental: combine both covariances
        # This gives a dim_in steering direction informed by both input stats and gradient curvature
        U_a, S_a, _ = svd(A)
        U_g, S_g, _ = svd(G)
        
        if rank:
            U_a = U_a[:, :rank]
            S_a = S_a[:rank]
            U_g = U_g[:, :rank]
            S_g = S_g[:rank]
        
        # Mean gradient direction (what we're steering toward)
        mean_grad = gradients.mean(0)  # [dim_out]
        
        # Project through output space first, then map to input space
        # This is a heuristic; exact KFAC would use (G ⊗ A)^-1
        g_transformed = U_g.T @ mean_grad  # [rank_g]
        
        # Map to input space (rough approx: assume some correlation between input/output)
        # Better approach would use actual W matrix
        a_weights = S_a / (S_a.sum() + 1e-8)  # normalize
        direction = U_a @ (a_weights * g_transformed.mean())  # broadcast to [dim_in]
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return direction


def kfac_contrastive_steering(
    activations_pos: Float[Tensor, "batch/2 dim_in"],
    activations_neg: Float[Tensor, "batch/2 dim_in"],
    gradients_pos: Float[Tensor, "batch/2 dim_out"],
    gradients_neg: Float[Tensor, "batch/2 dim_out"],
    lambda_reg: float = 1e-2,
    mode: Literal["diff", "pos", "neg"] = "diff",
    kfac_mode: Literal["input", "output", "product"] = "input",
    rank: int | None = None,
) -> Float[Tensor, "dim"]:
    """
    Contrastive KFAC steering: compute separate covariances for pos/neg, then combine.
    
    Args:
        activations_pos/neg: Input activations for positive/negative examples
        gradients_pos/neg: Gradients for positive/negative examples
        lambda_reg: Regularization
        mode: How to combine pos/neg directions
            - "diff": v_pos - v_neg (contrastive direction)
            - "pos": Just positive direction
            - "neg": Just negative direction
        kfac_mode: Which covariance to use (input/output/product)
        rank: Number of eigenvectors to keep
    
    Returns:
        Steering direction
    """
    v_pos = kfac_steering(activations_pos, gradients_pos, lambda_reg, kfac_mode, rank)
    v_neg = kfac_steering(activations_neg, gradients_neg, lambda_reg, kfac_mode, rank)
    
    if mode == "diff":
        return v_pos - v_neg
    elif mode == "pos":
        return v_pos
    elif mode == "neg":
        return v_neg
    else:
        raise ValueError(f"Unknown mode: {mode}")
