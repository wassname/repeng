import torch
import torch.nn as nn
from contextlib import contextmanager

# Base class for all PEFT adapter layers
from peft.tuners.tuners_utils import BaseTunerLayer

# Specific layer types for ROAD and IA3
try:
    from peft.tuners.road.layer import RoadLayer # 3 weeks old
    from peft.tuners.ia3.layer import IA3Layer
except ImportError:
    raise ImportError("Could not import PEFT ROAD/IA3 layers—please install 'peft'.")

@contextmanager
def AdapterSteer(
    model: nn.Module,
    coeff: float = 1.0,
    scale_param: str = 'theta',  # For ROAD: 'theta' (recommended) or 'alpha'; ignored for IA3
    adapter_name: str = "default",
):
    """
    A context manager to temporarily scale ROAD or IA3 adapter parameters by a coefficient.
    - For ROAD: Scales theta (angles, for directional/invertible steering) or alpha (magnitudes).
    - For IA3: Scales the ia3_l vector (activation scalers).

    Args:
      model:          The PEFT model with ROAD or IA3 adapters.
      coeff:          The scaling coefficient (default: 1.0).
      scale_param:    For ROAD: 'theta' or 'alpha' (default: 'theta'); ignored for IA3.
      adapter_name:   Name of the adapter to target (default: "default").

    Usage:
        # Contrastive training example
        with AdapterModifier(model, coeff=1.0, scale_param='theta'):
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
        with AdapterModifier(model, coeff=-1.0, scale_param='theta'):
            outputs = model(input_ids)
            loss = loss_fn(outputs, -labels)
            loss.backward()

    FIXME: update and remove old parts and docs one I've worked out what works and what doesn't.
    Context & Task: 
      - Goal: Build reversible/scalable PEFT adapter control for contrastive learning (e.g., "honest" vs "dishonest" steering).
      - Usage: Two-pass training with coeff=1.0 (normal) and coeff=-1.0 (inverted) to learn bidirectional representations.
      - Challenge: Which adapter type + scaling approach works best for invertible steering without breaking pretrained capabilities?
      - Candidates: ROAD (rotation-based, geometrically reversible), IA3 (element-wise scaling, simple inversion), LoRA (additive, may not invert cleanly).

    Decision Rationale:
      - ROAD (from peft.tuners.road): Forward applies element-wise rotations via cos/sin(theta) * alpha on grouped activations.
        - Theta (angles): Core param (zero-init for identity). Scaling multiplies angles linearly (for small values), coeff=-1 inverts rotation exactly (geometric inverse). Recommended for contrastive/invertible steering (~70% confidence, based on 2D rotation math and paper's angular focus).
        - Alpha (scales): Multiplicative gain on cos/sin (one-init). Scaling amplifies/suppresses effect without direction change; coeff=-1 reflects (not pure inverse). Use for intensity control if theta warps (e.g., large angles).
        - Why multiply? Matches forward (element-wise mul); add/shift offsets arbitrarily (distorts geometry). BNB (8/4-bit): Params fp32, compatible.
      - IA3 (from peft.tuners.ia3): Scales activations via ia3_l vector mul. Coeff directly amplifies/inhibits (coeff=-1 flips); simple/invertible (~80% confidence, per paper's mixed-batch design). Why multiply? Inherent to method.
      - Prioritization: ROAD theta for subspace rotations/composability (best for steering concepts); IA3 for global scaling/simplicity (fallback if rotations fail). Test both on contrastive loss to validate inversion quality.
    """
    if scale_param not in ['theta', 'alpha'] and scale_param != 'auto':  # 'auto' for detection
        raise ValueError("scale_param must be 'theta', 'alpha', or ignored for IA3.")

    # store originals here
    original_states = []

    try:
        # --- ENTER: find & modify all adapter layers ---
        for name, module in model.named_modules():
            if isinstance(module, (RoadLayer, IA3Layer)) and adapter_name in module.active_adapters:
                if isinstance(module, RoadLayer):
                    if scale_param == 'theta':
                        param = module.road_theta[adapter_name]
                    else:  # alpha
                        param = module.road_alpha[adapter_name]
                    orig = param.data.clone()
                    original_states.append((module, 'road', scale_param, orig))
                    param.data.mul_(coeff)
                elif isinstance(module, IA3Layer):
                    param = module.ia3_l[adapter_name]
                    orig = param.data.clone()
                    original_states.append((module, 'ia3', orig))
                    param.data.mul_(coeff)

        yield

    finally:
        # --- EXIT: restore originals ---
        for module, kind, *extra in original_states:
            if kind == 'road':
                sp = extra[0]
                if sp == 'theta':
                    module.road_theta[adapter_name].data.copy_(extra[1])
                else:
                    module.road_alpha[adapter_name].data.copy_(extra[1])
            elif kind == 'ia3':
                module.ia3_l[adapter_name].data.copy_(extra[0])
