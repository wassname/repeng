from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float


def compute_reprpo_nll_margin_loss(
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
