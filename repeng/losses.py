from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float


def compute_simpo_loss(logp_pos, logp_neg, **kwargs):
    """Compute the SimPO loss for a batch of positive and negative log probabilities.

    Why do we use SimPO loss? Unlike DPO it's closer to our autoregressive steering use case, so the curvature of the loss landscape is more similar.

    Args:
        logp_pos: Log probabilities of the positive responses. Shape: (batch_size/2,)
        logp_neg: Log probabilities of the negative responses. Shape: (batch_size/2,)
    See 
    - https://github.com/princeton-nlp/SimPO/blob/1b3e8f3528a23bce3da514a2dce8ea7490d4bc75/scripts/simpo_trainer.py#L560)
    - https://github.com/princeton-nlp/SimPO/blob/1b3e8f3528a23bce3da514a2dce8ea7490d4bc75/scripts/simpo_config.py#L56
    """
    pi_logratios = logp_pos - logp_neg
    gamma_beta_ratio = 0.25
    beta = 2.0
    label_smoothing = 0.0
    logits = pi_logratios - gamma_beta_ratio
    return (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    ).mean()

def compute_reprpo_loss(hs_pos: Float[Tensor, "batch/2 hidden_dim"],
                       hs_neg: Float[Tensor, "batch/2 hidden_dim"], **kwargs) -> Float[Tensor, ""]:
    """
    Compute loss that creates useful gradients for steering.
    
    Key: We want gradients that point in opposite directions for pos/neg,
    creating a clear preference axis.
    """
    # Current preference direction  
    pref_dir = hs_pos - hs_neg  # [batch/2, hidden_dim]
    
    # Option 1: Direct magnitude (simplest, often works best)
    pref_mag = torch.norm(pref_dir, dim=-1)  # [batch/2]
    loss = -pref_mag.mean()  # Negative because we maximize
    
    # Option 2: Log magnitude (more stable)
    # eps = 1e-8
    # loss = -torch.log(pref_mag + eps).mean()

    # Hinge loss: only optimize if below margin
    margin = 1
    loss = F.relu(margin - pref_mag).mean()
    
    return loss

def compute_cosine_loss(hs_pos,
                       hs_neg, **kwargs):
    """
    Maximize angle between pos and neg vectors.
    Makes them point in opposite directions.
    """
    # Normalize to unit vectors
    hs_pos_norm = F.normalize(hs_pos, dim=-1)
    hs_neg_norm = F.normalize(hs_neg, dim=-1)
    
    # Cosine similarity (want -1 for opposite directions)
    cos_sim = (hs_pos_norm * hs_neg_norm).sum(dim=-1)
    
    # Loss: we want cos_sim = -1
    loss = cos_sim.mean()  # Minimize this (want negative cosine)
    
    # Or push to specific angle (e.g., 120 degrees)
    # target_cos = -0.5  # cos(120°)
    # loss = F.mse_loss(cos_sim, torch.full_like(cos_sim, target_cos))
    
    return loss

def compute_kpo_loss(
    hs_pos: Float[Tensor, "batch/2 hidden_dim"],
    hs_neg: Float[Tensor, "batch/2 hidden_dim"],
    beta: float = 0.1,
    **kwargs
) -> Float[Tensor, ""]:
    """
    KPO: optimize relative advantage without reference.
    Based on prospect theory - losses loom larger than gains.
    """
    # Midpoint as implicit reference
    midpoint = (hs_pos + hs_neg) / 2
    
    # Distances from midpoint
    pos_advantage = torch.norm(hs_pos - midpoint, dim=-1)
    neg_disadvantage = torch.norm(hs_neg - midpoint, dim=-1)
    
    # Asymmetric weighting (losses hurt more)
    utility = pos_advantage - 2.0 * neg_disadvantage
    
    # Logistic to bounded range
    loss = -F.logsigmoid(beta * utility).mean()
    
    return loss

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
