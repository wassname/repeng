from jaxtyping import Float, Int
from torch import Tensor
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def safe_norm(x: Float[Tensor, "batch"], p: int = 2, dim: int = -1, eps: float = 1e-9):
    """
    Safe norm function to avoid division by zero.
    Returns a tensor with the same shape as x, where norms are clamped to eps.
    """
    norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    return x / (norm + eps)  # Avoid division by zero

def soft_clamp(x, min_val=-10.0, max_val=-0.01, sharpness=1.0):
    """
    Soft clamping using tanh - smoothly bounds values between min_val and max_val.
    sharpness controls how sharp the transition is (higher = sharper boundary).
    """
    center = (min_val + max_val) / 2
    range_half = (max_val - min_val) / 2
    return center + range_half * torch.tanh((x - center) / sharpness)

HS2 = Float[Tensor, "b h"]
HS = Float[Tensor, "b t h"]
Mask = Int[Tensor, "b t"]

def reduce_tokens_w_attention(
    x: HS, attn_mask: Mask,
    dim: int = 1,
) -> Float[Tensor, "b h"]:
    """mean of x, weighted by the attention mask, over dim (token or batch)
    with optional filtering of attention sinks"""
    
    # layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    
    return (x * attn_mask).sum(dim) / attn_mask.sum(dim)

def contrastive_steering_loss_with_ref(
    pref_dir_ref,
    hs_pi_pos,
    hs_pi_neg,
    ref_pos_label_logp,
    pi_pos_label_logp,
    cho_mask, 
    p=2,
    eps=1e-6,
    coef=1.0,
    margin=3.0,
    boundary_order=4,
    last_tokens_coherence=8,
    last_tokens_proj=8,
):
    """
    Contrastive loss for training reversible steering adapters.
    
    This loss trains an adapter to learn a steering direction that can be reversed
    by negating the coefficient. The adapter is applied with coef=1.0 for positive
    steering (e.g., honest) and coef=-1.0 for negative steering (e.g., dishonest).
    
    The loss has two components:
    1. Directional alignment: Maximizes projection onto reference direction when coef=1,
       minimizes when coef=-1 (this component reverses with coefficient)
    2. Coherence bounds: Ensures outputs remain coherent (doesn't reverse - always applied)
    
    Args:
        hs_ref_pos: Reference hidden states for positive examples (e.g., honest)
        hs_ref_neg: Reference hidden states for negative examples (e.g., dishonest)
        hs_pi_pos: Policy hidden states for positive examples (with adapter applied)
        hs_pi_neg: Policy hidden states for negative examples (with adapter applied)
        ref_pos_label_logp: Reference log probabilities for positive examples
        pi_pos_label_logp: Policy log probabilities for positive examples
        cho_mask: Attention mask for chosen sequences
        p: Norm order for normalization (default: 2 for L2)
        eps: Small epsilon for numerical stability
        coef: Coefficient indicating adapter direction (1.0 or -1.0)
              When training with AdapterSteer(model, coeff=coef), this should match
        margin: Margin for coherence constraint (default: 1.2)
    
    Returns:
        loss: Combined loss (directional + coherence)
        info: Dictionary with loss components for logging
    
    Training usage:
        for coef in [-1.0, 1.0]:
            with AdapterSteer(model, coeff=coef):
                outputs_pi = model(batch)
            loss, info = contrastive_steering_loss(..., coef=coef)
            loss.backward()
    """
    
    # Compute preference directions
    # pref_dir_ref = (hs_ref_pos - hs_ref_neg).detach()  # Reference direction (frozen)
    pref_dir_pi = hs_pi_pos - hs_pi_neg  # Policy direction (learnable via adapter)

    # Always treat as subspace (k, d); normalize rows to unit basis
    U = safe_norm(pref_dir_ref, p=p, dim=-1, eps=eps)  # (k, d)

    # Project the diff to subspace coords: pref_dir_pi @ U.T  (..., d) @ (d, k) -> (..., k)
    signed_proj = torch.einsum("...d,kd->...k", pref_dir_pi, U)  # (b t, k)

    # # Signed proj: Norm of projected diff (magnitude of separation in subspace)
    # # For signed alternative: .mean(dim=-1), but norm emphasizes strength
    # signed_proj = torch.norm(pref_proj, dim=-1)  # (b t,)

    # Scale projection by reference norm to get loss in predictable [0,2] range
    # When coef=1: maximize projection (minimize negative projection)
    # When coef=-1: minimize projection (maximize negative projection)
    ref_loss_hs_proj = torch.norm(U, dim=-1).sum() + 1  # Sum basis norms for multi
    loss_hs_proj = -signed_proj.max(-1)[0] / ref_loss_hs_proj  # scale loss as ratio
    loss_hs_proj = coef * loss_hs_proj  # Reverse loss direction based on intervention
    
    # Coherence constraint (doesn't reverse with coefficient - always enforced)
    baseline_logp = ref_pos_label_logp.detach()
    logp_pos = pi_pos_label_logp

    # Focus on suffix tokens (where the actual answer is)
    assert cho_mask[:, -2:].float().mean()==1, 'assume left padded'
    suffix_mask = cho_mask.clone()
    suffix_mask[:, :-8] = 0  # Focus on last 8 tokens while preserving padding info
    assert suffix_mask[:, -1].sum() > 0, "suffix_mask is all zero!"

    # Margin loss: allow up to 20% degradation in log probability, DPO often has similar nll degradation
    
    coherence_gap = (baseline_logp * margin - logp_pos)  # sequence-level constraint
    # coherence_gap = 
    
    # Soft clamp to prevent extreme values on one particular token, which is one way to game the loss
    coherence_gap = soft_clamp(coherence_gap, -10.0, 10.0, sharpness=1.0)
    
    # Quartic penalty for sharp boundary (consider reducing to quadratic for stability)
    loss_coherence_bounds = F.relu(coherence_gap)**boundary_order

    # Aggregate over tokens with attention weighting
    loss_coherence_bounds = reduce_tokens_w_attention(loss_coherence_bounds[-last_tokens_coherence:], cho_mask[:, :-1][-last_tokens_coherence:])

    # loss_hs_proj weighted mean with attn
    loss_hs_proj = reduce_tokens_w_attention(loss_hs_proj[-last_tokens_proj:], cho_mask[-last_tokens_proj:])

    # Combine losses
    loss = loss_hs_proj + loss_coherence_bounds

    assert torch.isfinite(loss).all(), "Non-finite loss encountered!"

    return loss, {
        "loss_hs_proj": loss_hs_proj,
        "loss_coherence_bounds": loss_coherence_bounds,
        "loss_total": loss,
        "dppx": (logp_pos - baseline_logp).mean(),
        "proj": signed_proj.mean() / coef,
    }



def contrastive_steering_loss_noref(
    hs_pos: Float[Tensor, "batch/2 hidden_dim"],
    hs_neg: Float[Tensor, "batch/2 hidden_dim"],
    logp_avg_pos_label: Float[Tensor, "batch/2"],  # Renamed for clarity: average logprob of positive sequence labels
    **kwargs
) -> Float[Tensor, ""]:
    """
    Maximize separation between positive and negative hidden states while enforcing a coherence constraint via a smooth quadratic margin.

    NOTE: This loss is NOT used for training—it's evaluated in a single forward/backward pass per sample to sample gradients and curvature for steering vector extraction. The goal is to position the loss computation such that both terms (separation and coherence) are active:
    - Separation: Encourages distinct hidden states for pos/neg pairs (via L2 norm of differences).
    - Coherence: Defines a bounded region where the NLL (negative log likelihood) of the preferred (positive) completion is no worse than a baseline, penalizing deviations quadratically outside this region.

    The baseline is the detached average logprob of positive labels, scaled by 0.99 to place the computation just inside the coherence boundary (ensuring the margin penalty contributes to gradients without dominating inside the region).

    Args:
        hs_pos: Hidden states from positive examples.
        hs_neg: Hidden states from negative examples.
        logp_avg_pos_label: Detached average logprob of labels for positive sequences (used as coherence baseline).

    Returns:
        Scalar loss value.
    """
    pref_dir = hs_pos - hs_neg
    pref_mag = torch.norm(pref_dir, dim=-1)

    # Detached baseline for coherence (no gradients flow through it)
    baseline_logp = logp_avg_pos_label.detach()

    # Quadratic penalty for coherence: Active outside the boundary (ReLU), grows quadratically. Scaled by 10000 to balance with separation term.
    # The 0.99 factor ensures we're just inside the boundary for gradient sampling.
    coherence_penalty = F.relu((baseline_logp * 0.99 - logp_avg_pos_label))**2 * 10000

    # Combined loss: Maximize separation (negative mean mag) + coherence penalty
    loss = -pref_mag.mean() + coherence_penalty.mean()

    return loss


