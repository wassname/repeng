import torch
import numpy as np
from torch.linalg import solve, qr
from jaxtyping import Float
from torch import Tensor
from typing import List, Dict, Optional, Literal
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .control import ControlModel


def natural_gradient_steering(
    grad_matrix: Float[Tensor, "batch dim"],  # gradients from SimPO loss
    lambda_reg: Optional[float] = 1e-2,  # regularization for Fisher inversion
    fim: Literal["empirical", "improved_empirical", "covariance"] = "improved_empirical",  # which FIM approximation
    low_dim: int | None = None,  # if set, project to random subspace of this dim
    seed: int | None = 42,  # for reproducible random projection
) -> Float[Tensor, "dim"]:
    """
    Computes natural gradient steering vector using Fisher preconditioning.
    Optionally projects to random subspace for large dims.
    
    Args:
        grad_matrix: Gradients from SimPO loss, shape (batch, dim)
        lambda_reg: Regularization for Fisher matrix inversion
        fim: Which FIM approximation to use ("empirical" or "covariance")
        low_dim: If set and < dim, project to random orthonormal subspace
        seed: Seed for random projection; None for non-reproducible
    
    Returns:
        Natural gradient steering vector
    """
    device = grad_matrix.device
    batch_size, dim = grad_matrix.shape

    # # HACK: Auto low-dim when severely undersampled
    # if batch_size < dim // 10:
    #     if low_dim is None or low_dim > batch_size // 2:
    #         low_dim = max(16, batch_size // 4)
    #         print(f"FIXME: Auto-setting low_dim={low_dim} for n={batch_size}, d={dim}")
    
    
    if low_dim and low_dim < dim:
        if seed is not None:
            torch.manual_seed(seed)
        # Random orthonormal projection matrix P [dim, low_dim]
        rand_matrix = torch.randn(dim, low_dim, device=device, dtype=torch.float32)
        rand_matrix = rand_matrix / torch.norm(rand_matrix, dim=0, keepdim=True)  # normalize columns
        P, _ = qr(rand_matrix, mode="reduced")  # orthonormal columns
        
        # Project gradients to subspace
        grad_matrix = grad_matrix @ P  # [batch, low_dim]
        dim = low_dim  # update for below
    
    # Mean gradient direction (what we want to steer towards)
    mean_grad = grad_matrix.mean(dim=0)  # [dim]
    
    match fim:
        case "empirical":
            # Empirical Fisher (EF): F ≈ (1/n) Σ g_i g_i^T
            # This is the second moment matrix of the gradients.
            # See Wu et al. (2024) https://arxiv.org/abs/2406.06420 for a discussion
            # of its limitations (inversely-scaled projection issue).
            F_matrix = torch.einsum('bi,bj->ij', grad_matrix, grad_matrix) / batch_size
        case "improved_empirical":
            # Improved Empirical Fisher (iEF) from Wu et al. (2024).
            # F_iEF = Σ_n (1 / ||∇_z(l_n)||^2) * g_n * g_n^T
            # This rescales each sample's contribution by its logit gradient norm
            # to address the EF's inversely-scaled projection issue.
            if logit_grad_norms is None:
                raise ValueError("logit_grad_norms must be provided when use_ief is True.")
            # Add a small epsilon to prevent division by zero for converged samples.
            weights = 1.0 / (logit_grad_norms.pow(2) + 1e-8)
            weighted_grads = grad_matrix * weights.unsqueeze(-1)
            F_matrix = torch.einsum('bi,bj->ij', weighted_grads, grad_matrix) / batch_size
        case "covariance":
            # Covariance of gradients: F ≈ Cov(g) = E[(g - E[g])(g - E[g])^T]
            # This is the Fisher for a Gaussian model, but an approximation otherwise.
            centered_grads = grad_matrix - mean_grad.unsqueeze(0)
            # Use n-1 for unbiased sample covariance estimate
            F_matrix = torch.einsum('bi,bj->ij', centered_grads, centered_grads) / (batch_size - 1)
        case _:
            raise ValueError(f"Unknown fim type: {fim}")

    # # HACK: Adaptive regularization based on trace (average eigenvalue)
    # if lambda_reg is None:
    #     trace_F = torch.trace(F_matrix)
    #     lambda_reg = max(
    #         1e-6, # minimum regularization
    #         0.01 * trace_F / dim,  # 1% of average eigenvalue
    #     )
    
    # Regularize for numerical stability
    F_reg = F_matrix + torch.eye(dim, device=device, dtype=torch.float32) * lambda_reg
    
    # Natural gradient: θ_natural = F^(-1) * ∇L
    v_natural = solve(F_reg, mean_grad)
    
    if low_dim:
        # HACK: Scale correction for low-dim projection
        # Preserve gradient magnitude scale
        orig_grad_norm = torch.norm(mean_grad)
        
        # Project back to original space
        v_natural = P @ v_natural  # [dim_orig, low_dim] @ [low_dim] -> [dim_orig]

        # Rescale to match original gradient scale
        proj_norm = torch.norm(v_natural)
        if proj_norm > 1e-8:
            v_natural = v_natural * (orig_grad_norm / proj_norm)
    
    
    return v_natural


