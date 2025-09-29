import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.road.layer import RoadLayer
from peft.tuners.ia3.layer import IA3Layer
from peft.tuners.lora.layer import LoraLayer


class AdapterScaler:
    """Handles scaling of adapter parameters during forward passes."""
    
    @staticmethod
    def scale_road_params(
        module: RoadLayer,
        args: Tuple,
        kwargs: Dict[str, Any],
        adapter_name: str,
        coeff: float,
        scale_param: str,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale ROAD adapter parameters."""
        param_attr = 'road_theta' if scale_param == 'theta' else 'road_alpha'
        param_dict = getattr(module, param_attr)
        
        if adapter_name in param_dict:
            original = param_dict[adapter_name]
            param_dict[adapter_name] = original * coeff
            originals.append((param_dict, adapter_name, original))
        
        return (args, kwargs)
    
    @staticmethod
    def scale_ia3_params(
        module: IA3Layer,
        args: Tuple,
        kwargs: Dict[str, Any],
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale IA3 adapter parameters."""
        param_dict = module.ia3_l
        
        if adapter_name in param_dict:
            original = param_dict[adapter_name]
            # Symmetric flip around 1.0: (original - 1) * coeff + 1
            scaled = (original - 1.0) * coeff + 1.0
            param_dict[adapter_name] = scaled
            originals.append((param_dict, adapter_name, original))
        
        return (args, kwargs)
    
    @staticmethod
    def scale_lora_params(
        module: LoraLayer,
        args: Tuple,
        kwargs: Dict[str, Any],
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale LoRA adapter parameters (A.weight and B.weight) for reversible steering."""
        # LoRA has lora_A and lora_B as ParameterDicts of nn.Linear modules
        if hasattr(module, 'lora_A') and adapter_name in module.lora_A:
            lora_A_linear = module.lora_A[adapter_name]
            original_weight_A = lora_A_linear.weight.data
            # Non-in-place: create new tensor
            lora_A_linear.weight.data = original_weight_A * coeff
            originals.append((lora_A_linear.weight, None, original_weight_A))
        
        if hasattr(module, 'lora_B') and adapter_name in module.lora_B:
            lora_B_linear = module.lora_B[adapter_name]
            original_weight_B = lora_B_linear.weight.data
            # Non-in-place: create new tensor
            lora_B_linear.weight.data = original_weight_B * coeff
            originals.append((lora_B_linear.weight, None, original_weight_B))
        
        return (args, kwargs)


@contextmanager
def AdapterSteer(
    model: nn.Module,
    coeff: float = 1.0,
    scale_param: str = 'theta',
    adapter_name: Optional[str] = None
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
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    
    if scale_param not in ['theta', 'alpha']:
        raise ValueError("scale_param must be 'theta' or 'alpha'.")
    
    if coeff is None:
        with model.disable_adapter(adapter_name):
            yield
        return
    
    hooks = []
    originals = []
    
    try:
        for name, module in model.named_modules():
            hook_fn = None
            
            if isinstance(module, RoadLayer) and adapter_name in module.active_adapters:
                hook_fn = partial(
                    AdapterScaler.scale_road_params,
                    adapter_name=adapter_name,
                    coeff=coeff,
                    scale_param=scale_param,
                    originals=originals
                )
            elif isinstance(module, IA3Layer) and adapter_name in module.active_adapters:
                hook_fn = partial(
                    AdapterScaler.scale_ia3_params,
                    adapter_name=adapter_name,
                    coeff=coeff,
                    originals=originals
                )
            elif isinstance(module, LoraLayer) and adapter_name in module.lora_A:  # Check lora_A as proxy
                hook_fn = partial(
                    AdapterScaler.scale_lora_params,
                    adapter_name=adapter_name,
                    coeff=coeff,
                    originals=originals
                )
            
            if hook_fn:
                try:
                    handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
                except TypeError:
                    handle = module.register_forward_pre_hook(hook_fn)
                hooks.append(handle)
        
        yield
        
    finally:
        # General restoration: (target_param, attr, original_value)
        # For dicts (IA3/ROAD): target is dict, attr is adapter_name str, original is tensor
        # For LoRA: target is Linear.weight tensor, attr is None, original is tensor
        for target, attr, original in originals:
            if attr is None:
                # Direct tensor restoration (LoRA weights)
                target.data = original
            else:
                # Dict restoration (IA3/ROAD)
                target[attr] = original
        
        for handle in hooks:
            handle.remove()


# ... rest of file unchanged
