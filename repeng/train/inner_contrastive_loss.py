from jaxtyping import Float, Int
from torch import Tensor
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

def safe_norm(x: Float[Tensor, "batch"], p: int = 2, dim: int = -1, eps: float = 1e-9):
    """
    Safe norm function to avoid division by zero.
    Returns a tensor with the same shape as x, where norms are clamped to eps.
    """
    norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    return x / (norm + eps)  # Avoid division by zero

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

# def select_top_k_directions(
#     pref_dir: Float[Tensor, "k d"],
#     hs_ref_cho: HS,
#     hs_ref_rej: HS,
#     cho_mask: Mask,
#     top_k: int = 2,
#     eps: float = 1e-6,
# ) -> Float[Tensor, "k_selected d"]:
#     """
#     Select top-k PCA directions that best align with reference separation.
    
#     Args:
#         pref_dir: All PCA directions (k, d)
#         hs_ref_cho/rej: Reference hidden states
#         cho_mask: Attention mask
#         top_k: Number of directions to keep
    
#     Returns:
#         Top-k directions (top_k, d)
#     """
#     # Reference separation vector
#     pref_dir_ref = hs_ref_cho - hs_ref_rej  # (b, t, d)
    
#     # Normalize directions
#     U = safe_norm(pref_dir, p=2, dim=-1, eps=eps)  # (k, d)
    
#     # Project ref separation onto all directions
#     proj_ref = torch.einsum("...d,kd->...k", pref_dir_ref, U)  # (b, t, k)
    
#     # Aggregate: mean absolute projection per direction
#     loss_mask = cho_mask[:, :-1] if cho_mask.shape[1] > pref_dir_ref.shape[1] else cho_mask
#     proj_mag_per_dir = reduce_tokens_w_attention(proj_ref.abs(), cho_mask).mean(0)  # (k,)
    
#     # Select top-k by magnitude
#     _, top_indices = torch.topk(proj_mag_per_dir, k=min(top_k, len(proj_mag_per_dir)))
    
#     return pref_dir[top_indices]  # (top_k, d)


def contrastive_steering_loss_with_ref(
    U: Float[Tensor, "k d"],
    hs_ref_cho: HS,
    hs_ref_rej: HS,
    hs_pi_pos: HS,
    hs_pi_neg: HS,
    ref_pos_label_logp: Float[Tensor, "b t"],
    pi_pos_label_logp: Float[Tensor, "b t"],
    cho_mask: Mask,
    p=2,
    eps=1e-6,
    coef=1.0,
    coherence_threshold=0.2,
    boundary_order=2,
    top_k_directions: int = 2,
):
    """
    Contrastive loss for reversible steering adapters.
    
    **Objective 1: Maximize separation** along frozen PCA direction U
    - Maximize |projection of (hs_pi_pos - hs_pi_neg) onto U|
    - Scaled by reference projection magnitude
    
    **Objective 2: Coherence constraint** (asymmetric margin in log-space)
    - Criterion: log p_pi(y|x) >= log p_ref(y|x) - threshold
    - Equivalently: p_pi >= exp(-threshold) * p_ref
    - Per-token ReLU margin prevents gaming via whitespace
    - Quadratic penalty (boundary_order=2) creates steep boundary
    
    Why log-space? Already have logprobs, numerically stable, threshold in nats is interpretable.
    Why per-token? Prevents model from making some tokens very likely to offset others becoming unlikely.
    Why asymmetric? Don't penalize improvements, only degradation beyond margin.
    
    Args:
        U: Frozen PCA directions (k, d)
        hs_ref_cho/rej: Reference model hidden states for chosen/rejected
        hs_pi_pos/neg: Adapter hidden states for positive/negative steering
        ref_pos_label_logp: Reference model next-token logprobs (b, t)
        pi_pos_label_logp: Adapter next-token logprobs (b, t)
        cho_mask: Attention mask (b, t)
        coef: Steering coefficient (+1 or -1)
        coherence_threshold: Max degradation in nats (e.g., 0.2 = can be ~18% worse in probability)
        boundary_order: Polynomial order for coherence penalty (default 2 = quadratic)
        top_k_directions: Number of PCA components to use (None = all)
    
    Returns:
        loss: scalar
        info: dict with loss components and monitoring metrics
    """
    loss_mask = cho_mask[:, :-1]  # For logprobs (align with shifted)
    hs_mask = cho_mask
    
    # Compute projections onto frozen PCA direction (all k directions initially)
    U = safe_norm(U, p=p, dim=-1, eps=eps)  # (k, d) normalized
    
    pref_dir_pi = hs_pi_pos - hs_pi_neg
    pref_dir_ref = hs_ref_cho - hs_ref_rej
    
    signed_proj_pi = torch.einsum("...d,kd->...k", pref_dir_pi, U)  # (b,t,k)
    signed_proj_ref = torch.einsum("...d,kd->...k", pref_dir_ref, U)
    
    # Select top-k directions per sample based on ref magnitude
    if top_k_directions is not None and top_k_directions < signed_proj_ref.shape[-1]:
        # Aggregate ref projection per direction: (b, t, k) -> (b, k)
        ref_proj_per_dir = reduce_tokens_w_attention(signed_proj_ref.abs(), hs_mask.unsqueeze(-1), dim=1)  # (b, t, k)
        
        # Get top-k indices per batch sample
        _, top_k_indices = torch.topk(ref_proj_per_dir, k=min(top_k_directions, signed_proj_ref.shape[-1]), dim=-1)  # (b, top_k)
        
        # Select top-k projections per sample (gather along k dimension)
        signed_proj_pi = torch.gather(signed_proj_pi, dim=-1, index=top_k_indices.unsqueeze(1).expand(-1, signed_proj_pi.shape[1], -1))  # (b, t, top_k)
        signed_proj_ref = torch.gather(signed_proj_ref, dim=-1, index=top_k_indices.unsqueeze(1).expand(-1, signed_proj_ref.shape[1], -1))  # (b, t, top_k)
    
    # Aggregate projections (mean over selected components, then masked mean over tokens)
    proj_pi = reduce(signed_proj_pi, 'b t k -> b t', 'mean')
    proj_ref = reduce(signed_proj_ref, 'b t k -> b t', 'mean')
    
    # Keep signed projections (don't use abs - direction matters!)
    proj_pi_signed = reduce_tokens_w_attention(proj_pi, hs_mask)  # (b,)
    proj_ref_signed = reduce_tokens_w_attention(proj_ref, hs_mask)  # (b,)
    
    # Projection loss: ratio of signed projections (coef flips direction)
    proj_ratio = proj_pi_signed / (proj_ref_signed.abs() + eps)  # (b,) can be negative
    proj_ratio_bounded = torch.tanh(proj_ratio)
    loss_proj = -proj_ratio_bounded  # Maximize absolute ratio
    loss_proj = coef * loss_proj
    loss_proj = loss_proj.mean()  # Average over batch


    
    # Coherence loss: penalize if logprob degrades beyond threshold
    # Criterion: log p_pi >= log p_ref - threshold  (in nats)
    ref_logp = ref_pos_label_logp.detach()
    pi_logp = pi_pos_label_logp
    
    # Degradation in log-space (positive = pi worse than ref)
    logp_degradation = (ref_logp - pi_logp).clamp(0, 100)  # (b, t), clamp to prevent overflow
    
    # Apply margin per-token (prevents gaming), then polynomial penalty
    margin_violation = F.relu(logp_degradation - coherence_threshold)  # Zero if within margin
    coherence_penalty_per_token = (margin_violation * 10) ** boundary_order  # Scale then polynomial (avoids shrinking small values)
    coherence_loss = reduce_tokens_w_attention(coherence_penalty_per_token, loss_mask).mean()  # scalar

    loss = loss_proj + coherence_loss

    assert torch.isfinite(loss).all(), "Non-finite loss"
    
    # Monitoring: average degradation (negative = improvement)
    avg_logp_degradation = reduce_tokens_w_attention(logp_degradation, loss_mask).mean()
    
    return loss, {
        "loss_proj": loss_proj,
        "loss_coherence": coherence_loss,
        "loss_total": loss,
        "proj_ratio": proj_ratio.mean(),
        "logp_degradation": avg_logp_degradation,  # nats (positive = worse)
        "prob_ratio": torch.exp(-avg_logp_degradation),  # p_pi/p_ref (monitoring only)
        "proj_pi_signed": proj_pi_signed.mean(),
        "proj_ref_signed": proj_ref_signed.mean(),
    }


def contrastive_steering_loss_noref(
    hs_pos: Float[Tensor, "batch/2 hidden_dim"],
    hs_neg: Float[Tensor, "batch/2 hidden_dim"],
    logp_avg_pos_label: Float[Tensor, "batch/2"],
    **kwargs
) -> Float[Tensor, ""]:
    """
    Loss for steering vector extraction (not training).
    Maximizes ||hs_pos - hs_neg|| with coherence constraint.
    Evaluated in single pass to sample gradients/curvature.
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


def contrastive_steering_loss_with_ref_uspace(
    U_pca: Float[Tensor, "k d"],  # Frozen PCA directions in U-space
    U_svd: Float[Tensor, "d r"],  # Layer's output singular vectors
    hs_ref_cho: HS,
    hs_ref_rej: HS,
    hs_pi_pos: HS,
    hs_pi_neg: HS,
    ref_pos_label_logp: Float[Tensor, "b t"],
    pi_pos_label_logp: Float[Tensor, "b t"],
    cho_mask: Mask,
    p=2,
    eps=1e-6,
    coef=1.0,
    coherence_threshold=0.2,
    boundary_order=2,
    top_k_directions: int = 2,
):
    """
    Modified contrastive loss in layer's U-space (singular vector basis).
    
    1. Project all HS to U-space: hs_u = hs @ U_svd  (d -> r)
    2. Compute differences in U-space
    3. Project onto frozen PCA direction (extracted in U-space)
    """
    # Project to U-space (r << d typically)
    hs_ref_cho_u = hs_ref_cho @ U_svd
    hs_ref_rej_u = hs_ref_rej @ U_svd
    hs_pi_pos_u = hs_pi_pos @ U_svd
    hs_pi_neg_u = hs_pi_neg @ U_svd
    
    # Now proceed as before, but in U-space
    return contrastive_steering_loss_with_ref(
        U=U_pca,
        hs_ref_cho=hs_ref_cho_u,
        hs_ref_rej=hs_ref_rej_u,
        hs_pi_pos=hs_pi_pos_u,
        hs_pi_neg=hs_pi_neg_u,
        ref_pos_label_logp=ref_pos_label_logp,
        pi_pos_label_logp=pi_pos_label_logp,
        cho_mask=cho_mask,
        p=p,
        eps=eps,
        coef=coef,
        coherence_threshold=coherence_threshold,
        boundary_order=boundary_order,
        top_k_directions=top_k_directions,
    )


