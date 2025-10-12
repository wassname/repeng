# Loss Function Geometry

## Contrastive Steering Loss (`contrastive_steering_loss_noref`)

This diagram illustrates the geometric intuition behind the steering vector extraction loss.

![Loss Geometry](loss.svg)

### Loss Components

The loss has two terms that create this geometry:

1. **Separation Term** (maximize projection):
   ```python
   pref_dir = hs_pos - hs_neg  # Δact in diagram
   pref_mag = torch.norm(pref_dir, dim=-1)  # ||Δact||
   loss_separation = -pref_mag.mean()  # Maximize magnitude
   ```
   
   Pushes the current sample vector away from origin, increasing separation between positive and negative activations.

2. **Coherence Penalty** (stay inside boundary):
   ```python
   baseline_logp = logp_avg_pos_label.detach()  # Reference perplexity
   coherence_penalty = F.relu((baseline_logp * 0.99 - logp_avg_pos_label))**2 * 10000
   ```
   
   Creates a soft boundary in activation space: samples that degrade next-token logprob beyond 99% of baseline incur quadratic penalty. The 0.99 factor ensures we're just inside the boundary, so both terms contribute gradients.

### Key Properties

- **Space**: High-dimensional activation space where each point is a difference vector `act_pos - act_neg` (contrastive pair)
- **PCA Direction**: Frozen reference direction from base model (or coeff=0), prevents gaming by rotation
- **Coherence Boundary**: Not strictly spherical (depends on logprob landscape), but conceptually a region of "valid" activations that don't degrade generation quality
- **Optimization**: Gradients push samples toward maximum projection onto PCA direction while staying coherent

### Why This Works

The frozen PCA direction captures the concept axis (e.g., honest vs dishonest) from base model statistics. By maximizing projection onto this axis *while maintaining coherence*, we extract gradients that point toward "more honest" directions without breaking the model's generation capabilities.

The 10000x weighting balances the two terms (separation is typically small L2 norms, coherence is logprob scale).
