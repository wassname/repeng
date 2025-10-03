"""
Adapter steering for contrastive training with proper gradient flow.

Why temporarily replace ParameterDict with dict of scaled tensors?

We need to satisfy 3 competing requirements:
1. Optimizer tracks original Parameter objects - can't replace them
2. Gradients flow through differentiable operations - need param * coeff in graph  
3. Forward pass uses scaled values - module.vera_lambda_b[adapter_name] must return scaled tensor

"""
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.ia3.layer import IA3Layer
from peft.tuners.vera.layer import VeraLayer
from peft.tuners.road.layer import RoadLayer
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
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale IA3 adapter parameters."""
        # Store original ParameterDict
        originals.append((module, 'ia3_l', module.ia3_l))
        
        # Replace with new dict containing scaled tensors
        # Use object.__setattr__ to bypass nn.Module's type checking
        # Gradients flow through multiplication to original parameters
        object.__setattr__(module, 'ia3_l', {
            k: ((v - 1.0) * coeff + 1.0) if k == adapter_name else v
            for k, v in module.ia3_l.items()
        })
    
    @staticmethod
    def scale_vera_params(
        module: VeraLayer,
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale VeRA adapter parameters (lambda_d and lambda_b) for reversible steering.
        
        Scales both lambda_d and lambda_b by coeff. Since they multiply together in the
        forward pass, this gives an overall scaling of coeff^2, but allows both parameters
        to learn the steering direction. Gradients flow back through the multiplication
        to the original Parameter objects tracked by the optimizer.
        """

        # TODO not sure if I should do this param too
        # if hasattr(module, 'vera_lambda_d') and adapter_name in module.vera_lambda_d:
        #     # Store original ParameterDict
        #     originals.append((module, 'vera_lambda_d', module.vera_lambda_d))
            
        #     # Replace with new dict containing scaled tensors
        #     # Use object.__setattr__ to bypass nn.Module's type checking
        #     object.__setattr__(module, 'vera_lambda_d', {
        #         k: v * coeff if k == adapter_name else v
        #         for k, v in module.vera_lambda_d.items()
        #     })
        
        if hasattr(module, 'vera_lambda_b') and adapter_name in module.vera_lambda_b:
            # Store original ParameterDict
            originals.append((module, 'vera_lambda_b', module.vera_lambda_b))
            
            # Replace with new dict containing scaled tensors
            # Computation graph: scaled_tensor = original_param * coeff
            # Gradients will flow back through multiplication to original_param
            object.__setattr__(module, 'vera_lambda_b', {
                k: v * coeff if k == adapter_name else v
                for k, v in module.vera_lambda_b.items()
            })
        

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
    adapter_name: Optional[str] = None
):
    """
    A context manager to temporarily scale IA3 or VeRA adapter parameters by a coefficient.
    - For IA3: Scales the ia3_l vector (activation scalers) symmetrically around 1.0.
    - For VeRA: Scales the lambda_d and lambda_b vectors (input/output scalers).

    Args:
      model:          The PEFT model with IA3 or VeRA adapters.
      coeff:          The scaling coefficient (default: 1.0).
      adapter_name:   Name of the adapter to target (default: model's active_adapter).

    Usage:
        # Contrastive training example
        with AdapterSteer(model, coeff=1.0):
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
        with AdapterSteer(model, coeff=-1.0):
            outputs = model(input_ids)
            loss = loss_fn(outputs, -labels)
            loss.backward()

    Context & Task: 
      - Goal: Build reversible/scalable PEFT adapter control for contrastive learning (e.g., "honest" vs "dishonest" steering).
      - Usage: Two-pass training with coeff=1.0 (normal) and coeff=-1.0 (inverted) to learn bidirectional representations.
      - Challenge: Focus on multiplicative/scaling adapters (IA3, VeRA) for clean invertibility without breaking pretrained capabilities.
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    
    if coeff is None:
        with model.disable_adapter():
            yield
        return
    
    hooks = []
    originals = []
    
    try:
        for name, module in model.named_modules():
            if isinstance(module, IA3Layer) and adapter_name in module.active_adapters:
                AdapterScaler.scale_ia3_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
            elif isinstance(module, VeraLayer) and adapter_name in module.vera_lambda_b:
                AdapterScaler.scale_vera_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
        
        yield
        
    finally:
        # Restore original ParameterDicts before backward pass
        # Format: (module, attr_name, original_param_dict)
        for module, attr_name, original_param_dict in originals:
            setattr(module, attr_name, original_param_dict)


