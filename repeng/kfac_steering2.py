"""
KFAC-style steering using Fisher Information Matrix approximations.

Based on Kronecker-Factored Approximate Curvature (KFAC):
- Martens & Grosse 2015: https://arxiv.org/abs/1503.05671
- "From Memorization to Reasoning" 2024: https://arxiv.org/html/2510.24256v2

Key KFAC insight: For layer with linear transform y = W @ x,
the Fisher Information Matrix can be approximated as:
    F_W ≈ G ⊗ A
where:
- A = E[x x^T]: input activation covariance [dim_in × dim_in]
- G = E[g g^T]: output gradient covariance [dim_out × dim_out]
- g = ∂loss/∂y: gradients w.r.t. layer outputs

The paper shows that:
- Top eigenvectors of G ⊗ A = high-curvature directions = shared/generalizable features
- Bottom eigenvectors = low-curvature = memorization/narrow features

For steering: We want to align with top eigenvectors (high curvature = important directions)
weighted by contrastive signal (positive - negative examples).

Usage: Same interface as original kfac_steering.py, but with corrected math.
"""
import torch
from jaxtyping import Float
from torch import Tensor
from typing import Literal


def kfac_steering(
    activations: Float[Tensor, "batch dim"],
    gradients: Float[Tensor, "batch dim"],
    lambda_reg: float = 1e-2,
    mode: Literal["natural_grad", "pca", "gradient_pca"] = "natural_grad",
    rank: int | None = None,
) -> Float[Tensor, "dim"]:
    """
    Compute KFAC-informed steering direction from activations and gradients.
    
    This computes a steering vector in activation space that can be added to layer outputs
    in a post-hook, similar to standard PCA steering but using curvature information.
    
    Theory:
    - "natural_grad": Precondition activation difference with Fisher inverse (A^{-1} @ Δh)
      → Steers along natural gradient (accounts for parameter curvature)
    - "pca": Top eigenvector of activation covariance (A)
      → High-variance direction (like PCA on h_pos - h_neg but curvature-weighted)
    - "gradient_pca": Top eigenvector of gradient covariance (G)
      → High-curvature direction (what the loss cares about most)
    
    Args:
        activations: Layer activations [batch, dim] (e.g., hidden states at layer L)
        gradients: Gradients ∂loss/∂activations [batch, dim] from ReprPO loss
        lambda_reg: Tikhonov regularization for covariance matrices
        mode: Which KFAC-based direction to compute
        rank: Optional rank truncation (keep top-k eigenvectors)
    
    Returns:
        Steering direction [dim] normalized to unit L2 norm
    """
    device = activations.device
    batch_size = activations.shape[0]
    dim = activations.shape[1]
    
    # Compute uncentered covariance (second moment) as per KFAC
    # FIXED: Use h.T @ h (not einsum with wrong indices)
    # A = E[h h^T] where h are activations
    A = (activations.T @ activations) / batch_size  # [dim, dim]
    A = A + lambda_reg * torch.eye(dim, device=device, dtype=A.dtype)
    
    # G = E[g g^T] where g are gradients
    G = (gradients.T @ gradients) / batch_size  # [dim, dim]
    G = G + lambda_reg * torch.eye(dim, device=device, dtype=G.dtype)
    
    if mode == "natural_grad":
        # Natural gradient: precondition with A^{-1} (Fisher-informed direction)
        # This is the "right" direction to move in parameter space accounting for curvature
        # For steering: direction = A^{-1} @ mean(activations)
        # Can also use on contrastive diff: A^{-1} @ (mean(h_pos) - mean(h_neg))
        try:
            A_inv = torch.linalg.inv(A)
        except torch.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            A_inv = torch.linalg.pinv(A)
        
        mean_activation = activations.mean(0)  # [dim]
        direction = A_inv @ mean_activation
        
    elif mode == "pca":
        # Top eigenvector of activation covariance (like PCA but on A, not centered)
        # These are the "important" directions in activation space (high variance)
        # Paper: top eigenvectors = generalizing directions
        eigenvals, eigenvecs = torch.linalg.eigh(A)  # Sorted ascending
        
        # Take top-k (largest eigenvalues)
        k = rank if rank is not None else dim
        k = min(k, dim)
        
        top_eigenvecs = eigenvecs[:, -k:]  # [dim, k] (last k = largest)
        top_eigenvals = eigenvals[-k:]     # [k]
        
        # Project mean activation onto top eigenvectors, weight by eigenvalues
        mean_activation = activations.mean(0)  # [dim]
        projection = top_eigenvecs.T @ mean_activation  # [k]
        weighted_proj = projection * torch.sqrt(top_eigenvals)  # Weight by sqrt(λ) for smoother scaling
        direction = top_eigenvecs @ weighted_proj  # [dim]
        
    elif mode == "gradient_pca":
        # Top eigenvector of gradient covariance (high-curvature directions)
        # These are directions where the loss changes most (high Fisher information)
        # Paper: top G eigenvectors = directions model uses for shared structure
        eigenvals, eigenvecs = torch.linalg.eigh(G)
        
        k = rank if rank is not None else dim
        k = min(k, dim)
        
        top_eigenvecs = eigenvecs[:, -k:]
        top_eigenvals = eigenvals[-k:]
        
        mean_gradient = gradients.mean(0)  # [dim]
        projection = top_eigenvecs.T @ mean_gradient  # [k]
        weighted_proj = projection * torch.sqrt(top_eigenvals)
        direction = top_eigenvecs @ weighted_proj
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'natural_grad', 'pca', or 'gradient_pca'")
    
    # L2 normalize (critical for stable steering magnitudes)
    direction = direction / (torch.norm(direction) + 1e-8)
    
    return direction


def kfac_contrastive_steering(
    activations_pos: Float[Tensor, "batch/2 dim"],
    activations_neg: Float[Tensor, "batch/2 dim"],
    gradients_pos: Float[Tensor, "batch/2 dim"],
    gradients_neg: Float[Tensor, "batch/2 dim"],
    lambda_reg: float = 1e-2,
    mode: Literal["diff", "pos", "neg"] = "diff",
    kfac_mode: Literal["natural_grad", "pca", "gradient_pca"] = "natural_grad",
    rank: int | None = None,
) -> Float[Tensor, "dim"]:
    """
    Contrastive KFAC steering: compute directions for pos/neg, then combine.
    
    This is the main function for extracting steering vectors from paired examples.
    Like PCA(h_pos - h_neg), but using KFAC curvature information.
    
    Compatible interface with original kfac_steering.py kfac_contrastive_steering().
    
    Args:
        activations_pos/neg: Activations for positive/negative examples
        gradients_pos/neg: Gradients for positive/negative examples
        lambda_reg: Regularization
        mode: How to combine pos/neg directions
            - "diff": v_pos - v_neg (contrastive direction) **recommended**
            - "pos": Just positive direction
            - "neg": Just negative direction
        kfac_mode: Which KFAC method to use (see kfac_steering docs)
        rank: Number of eigenvectors to keep
    
    Returns:
        Steering direction [dim] (unit normalized, pointing from neg to pos)
    """
    if mode == "diff":
        # Recommended: Compute direction on combined data, then weight by contrastive signal
        # This pools Fisher information from both pos and neg
        h_all = torch.cat([activations_pos, activations_neg], dim=0)
        g_all = torch.cat([gradients_pos, gradients_neg], dim=0)
        
        # Compute KFAC direction on pooled data
        direction = kfac_steering(h_all, g_all, lambda_reg, kfac_mode, rank)
        
        # Sign-align: positive examples should move in +direction
        delta_h = activations_pos.mean(0) - activations_neg.mean(0)
        if direction @ delta_h < 0:
            direction = -direction
            
    elif mode == "pos":
        # Direction from positive examples only
        direction = kfac_steering(activations_pos, gradients_pos, lambda_reg, kfac_mode, rank)
        
    elif mode == "neg":
        # Direction from negative examples only (then flip)
        direction = kfac_steering(activations_neg, gradients_neg, lambda_reg, kfac_mode, rank)
        direction = -direction  # Flip to point away from negative
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return direction
