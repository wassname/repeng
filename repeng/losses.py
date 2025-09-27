from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float


def compute_reprpo_nll_margin_loss(
    hs_pos: Float[Tensor, "batch/2 hidden_dim"],
    hs_neg: Float[Tensor, "batch/2 hidden_dim"],
    logp_pos: Float[Tensor, "batch/2"],
    **kwargs
) -> Float[Tensor, ""]:
    """
    Maximize separation with coherence constraint via smooth switching.    
    """
    pref_dir = hs_pos - hs_neg
    pref_mag = torch.norm(pref_dir, dim=-1)
    
    # Use current as baseline (no hyperparams)
    baseline_logp = logp_pos.detach()
    
    # Smooth switching: ReLU gives gradients when active
    # coherence_penalty = F.relu(100 * (baseline_logp*0.99 - logp_pos)) ** 2   # Large Penalty if degrading
    # coherence_penalty = F.relu(100 * (baseline_logp*1.1 - logp_pos)) ** 2 
    coherence_penalty = F.relu((baseline_logp*0.99 - logp_pos))**2 * 10000
    # coherence_penalty = F.softplus(100 * (baseline_logp - logp_pos)) * 100
    # coherence_penalty = (baseline_logp - logp_pos) ** 2 * 1000
    
    # 1st order hidden state distance vs 2nd order coherence boundary
    loss = -pref_mag.mean() + coherence_penalty.mean()
    
    return loss
