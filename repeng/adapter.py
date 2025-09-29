import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.ia3.layer import IA3Layer
from peft.tuners.vera.layer import VeraLayer


class AdapterScaler:
    """Handles scaling of adapter parameters during forward passes."""
    
    @staticmethod
    def scale_ia3_params(
        module: IA3Layer,
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
            # scaled = original * coeff  # Direct mul: inverts for -1, amplifies for +1
            param_dict[adapter_name] = scaled
            originals.append((param_dict, adapter_name, original))
        
    
    @staticmethod
    def scale_vera_params(
        module: VeraLayer,
        adapter_name: str,
        coeff: float,
        originals: List[Tuple]
    ) -> Tuple:
        """Scale VeRA adapter parameters (lambda_d and lambda_b) for reversible steering."""
        if hasattr(module, 'vera_lambda_d') and adapter_name in module.vera_lambda_d:
            lambda_d = module.vera_lambda_d[adapter_name]
            original_d = lambda_d
            module.vera_lambda_d[adapter_name] = lambda_d * coeff
            originals.append((module.vera_lambda_d, adapter_name, original_d))
        
        if hasattr(module, 'vera_lambda_b') and adapter_name in module.vera_lambda_b:
            lambda_b = module.vera_lambda_b[adapter_name]
            original_b = lambda_b
            module.vera_lambda_b[adapter_name] = lambda_b * coeff
            originals.append((module.vera_lambda_b, adapter_name, original_b))
        


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
                AdapterScaler(scale_ia3_params=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
            elif isinstance(module, VeraLayer) and adapter_name in module.vera_lambda_d:
                AdapterScaler.scale_vera_params(module=module, adapter_name=adapter_name, coeff=coeff, originals=originals)
        
        yield
        
    finally:
        # General restoration: (target_param, attr, original_value)
        # For dicts (IA3/VeRA): target is dict, attr is adapter_name str, original is tensor
        for target, attr, original in originals:
            # Dict restoration (IA3/VeRA)
            target[attr] = original
        

