"""
TRM SVFT adapter - combines SVFT (Singular Value Fine-Tuning) with changes

SVFT decomposes weights via SVD: W = U @ S @ V^T
- U, V are frozen singular vectors (orthonormal bases)
- S is diagonal singular values (frozen as s0)
- dS is sparse learnable delta to S (controlled by gate)

Changes are
- Only diagonal
- Add a tail instead of discarding tail of singular vector
- learnable decoder U via delta parameterization (U_eff = U_init + U_delta) which allows the model to modify learned direction which increase expressivity
- SVFT modes: replace_add, replace_mul, adapter_add, **adapter_mult**
- bounded singular values for stability, as negative singular values cause issues
- modified SVD equation to stay in low rank space. Instead of `(U @ S @ V^T) @ x`, we do `(x @ V.T) @ S @ U.T`

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import repeat, rearrange
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner
from peft.config import PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType
from peft.utils.other import get_pattern_key
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit, Int8Params

from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)



@dataclass
class TRMSvftAConfig(PeftConfig):
    """
    Configuration for TRM SVFT adapter.

    Config from https://github.com/VijayLingam95/SVFT/blob/8303115d45868712f952e6a847735bb59b1a9f18/MetaMath/run_math.sh#L29
    
    Hybrid SVD merging: Approximate full SVD cheaply. Principal (top-principal_rank SVD) captures base variance. Tail (low-rank random ortho basis to principal V, zero S init) merges tail info without full compute. Hypothesis: Principal leverages pretrain; tail recovers subtle patterns > pure top-k or random LoRA. Concat bases; single TRM on r=principal+tail (principal strong init, tail explores).
    """
    # SVFT-specific parameters
    r: int = field(default=19, metadata={"help": "Rank, includes Top-k SVD rank for principal directions (base variance), and tail_rank for low-rank approx of remaining vectors (subtle info)"})
    tail_rank: int = field(default=4, metadata={"help": "Low-rank approx rank for tail merging (ortho random basis; subtle info)"})
    # NOTE: off_diag disabled - diagonal-only (Plain SVFT) for simplicity and parameter efficiency
    # Paper shows full-rank diagonal outperforms low-rank banded for same param count
    fill_orthonormal: bool = field(
        default=False, 
        metadata={"help": "Fill beyond r with random orthonormal (disabled; tail replaces it)"}
    )
    learnable_u: bool = field(
        default=True,
        metadata={"help": "Make U learnable via delta parameterization (U_eff = U_init + U_delta). Weight decay pulls U_delta→0."}
    )
    svft_mode: Literal["replace_add", "replace_mul", "adapter_add", "adapter_mult"] = field(
        default="adapter_add",
        metadata={
            "help": "SVFT mode: replace_add (s0+sd, replace base), replace_mul (s0*(1+sd), replace base), adapter_add (sd only, add to base), adapter_mult (sd*s0 only, add to base)"
        }
    )
    svft_coeff: float = field(
        default=1.0,
        metadata={"help": "Steering strength multiplier. Can be negative to invert learned direction for mult modes or zero to disable."}
    )
    
    # Standard PEFT parameters
    target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply adapter to"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules to save (not adapt)"}
    )

    def __post_init__(self):
        self.peft_type = 'TRMSVFT'
        assert self.r > self.tail_rank, "Total rank r must be greater than tail_rank"
        self.principal_rank = self.r - self.tail_rank
        # self.r = self.principal_rank + self.tail_rank  # Total rank
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class TRMSvftLayer(BaseTunerLayer):
    """
    TRM SVFT layer that wraps a base layer and applies TRM-enhanced SVFT.
    
    SVFT decomposes W = U @ S @ V^T where:
    - U, V are frozen orthonormal bases from SVD
    - S = s0 + sd where s0 is frozen diagonal, sd is diagonal learnable delta
    - TRM recursively refines sd in r-dimensional singular value space
    
    NOTE: Currently diagonal-only (Plain SVFT). Off-diagonal variants disabled for simplicity.
    Paper shows full-rank diagonal outperforms low-rank banded at same parameter count.

    Code from https://github.com/VijayLingam95/SVFT/blob/8303115d45868712f952e6a847735bb59b1a9f18/svft/svft_layers.py
    """

    adapter_layer_names = ("svft_u_delta", "svft_dS")
    other_param_names = ("svft_u_base", "svft_down_proj", "svft_up_proj", "svft_s0", "svft_mode", "svft_coeff", "r", "tail_rank", "fill_orthonormal", "learnable_u")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # BaseTunerLayer.__init__(self, base_layer)

        self.r = {}
        self.tail_rank = {}
        self.fill_orthonormal = {}
        self.learnable_u = {}
        
        # SVFT components (per adapter)
        # PiSSA-style: store V.T @ sqrt(S) and U @ sqrt(S) for symmetric down/up projection
        self.svft_u_base = BufferDict({})  # Frozen U (unscaled, for learnable_u delta)
        self.svft_u_delta = nn.ParameterDict({})  # Learnable delta: U_eff = U_base + U_delta
        self.svft_down_proj = BufferDict({})  # V.T @ sqrt(S0): [d_in, r]
        self.svft_up_proj = BufferDict({})  # U @ sqrt(S0): [d_out, r] (before delta)
        self.svft_s0 = BufferDict({})  # Original singular values (for dS scaling)
        self.svft_dS = nn.ParameterDict({})  # Learnable delta to singular values

        # TRM components (single for combined r)
        self.svft_mode: Dict[str, str] = {}  # Per-adapter SVFT mode
        self.svft_coeff: Dict[str, float] = {}  # Per-adapter steering coefficient

        # Mark the weight as unmerged
        self._disable_adapters = False

        # Marker for Coconut to find TRM layers
        self._recursion_cache = None

        self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        svft_mode,
        svft_coeff,
        r,
        tail_rank,
        fill_orthonormal,
        learnable_u,
        **kwargs
    ) -> None:
        """
        Initialize SVFT adapter on this layer with hybrid SVD merging (concat principal + tail bases).
        """
        if adapter_name in self.svft_down_proj:
            return  # Already initialized

        self.svft_mode[adapter_name] = svft_mode
        self.svft_coeff[adapter_name] = float(svft_coeff)
        self.r[adapter_name] = r
        self.tail_rank[adapter_name] = tail_rank
        self.fill_orthonormal[adapter_name] = fill_orthonormal
        self.learnable_u[adapter_name] = learnable_u

        # Compute SVD of base weight
        base_weight = self.get_base_layer().weight
        
        # Dequantize if needed for full-precision SVD
        if isinstance(base_weight, Params4bit):
            base_weight = bnb.functional.dequantize_4bit(base_weight.data, base_weight.quant_state)
        elif isinstance(base_weight, Int8Params):
            base_weight = bnb.functional.dequantize_8bit(base_weight.data, base_weight.quant_state)
        
        base_weight = base_weight.float()  # [out, in]
        device = base_weight.device

        # principal_r = self.principal_rank
        # tail_rank = self.tail_rank
        # r = self.r
        principal_r = r - tail_rank

        # Compute full SVD first (before truncation)
        U_full, S_full, Vh_full = torch.linalg.svd(base_weight, full_matrices=False)
        
        # Principal: top-principal_rank SVD components
        U_p = U_full[:, :principal_r]
        Vh_p = Vh_full[:principal_r, :]
        S_p = S_full[:principal_r]
        
        # Tail: extract actual discarded singular vectors from full SVD
        # Then compress to tail_rank via random projection
        remaining = min(base_weight.shape) - principal_r  # Available tail dimensions
        if tail_rank > 0 and remaining > 0:
            # Get all tail singular vectors (the ones we'd normally discard)
            U_tail_full = U_full[:, principal_r:]  # [out, remaining]
            Vh_tail_full = Vh_full[principal_r:, :]  # [remaining, in]
            S_tail_full = S_full[principal_r:]  # [remaining]

            # Compress tail subspace to tail_rank dimensions via random projection
            # This preserves the tail subspace but mixes all tail directions
            actual_tail_rank = min(tail_rank, remaining)  # Can't exceed available dims
            
            # Random projection matrix [remaining, actual_tail_rank]
            proj_matrix = torch.randn(remaining, actual_tail_rank, device=device)
            proj_matrix, _ = torch.linalg.qr(proj_matrix)  # Orthonormalize
            
            U_tail = U_tail_full @ proj_matrix  # [out, actual_tail_rank]
            Vh_tail = (proj_matrix.T @ Vh_tail_full)  # [actual_tail_rank, in]
            # S_tail: weighted combination of tail singular values
            S_tail = (S_tail_full.unsqueeze(1) * proj_matrix).norm(dim=0)  # [actual_tail_rank]
            # Ensure non-zero for numerical stability
            S_tail = torch.clamp(S_tail, min=1e-6)
        else:
            # No tail: either tail_rank=0 or no remaining dimensions
            U_tail = None
            Vh_tail = None
            S_tail = None
        
        # Concat principal + tail for combined basis
        if U_tail is not None:
            U = torch.cat([U_p, U_tail], dim=1)
            Vh = torch.cat([Vh_p, Vh_tail], dim=0)
            S = torch.cat([S_p, S_tail])
        else:
            U = U_p
            Vh = Vh_p
            S = S_p
        
        # # Optionally fill remaining with orthonormal (if r < full)
        # full_min = min(base_weight.shape)
        # if self.fill_orthonormal and r < full_min:
        #     diff_rank = full_min - r
        #     U_fill = torch.randn(base_weight.shape[0], diff_rank, device=device)
        #     nn.init.orthogonal_(U_fill)
        #     Vh_fill = torch.randn(diff_rank, base_weight.shape[1], device=device)
        #     nn.init.orthogonal_(Vh_fill)
        #     U = torch.cat([U, U_fill], dim=1)
        #     Vh = torch.cat([Vh, Vh_fill], dim=0)
        #     S = torch.cat([S, torch.zeros(diff_rank, device=device)])
        #     r = S.shape[0]
        
        # Store combined U, Vh, S with PiSSA-style sqrt(S) pre-multiplication
        sqrt_S = torch.sqrt(S.clone().detach())  # [r]
        
        self.svft_u_base[adapter_name] = U.clone().detach().contiguous()  # [d_out, r] - unscaled U
        self.svft_u_delta[adapter_name] = nn.Parameter(
            torch.zeros_like(U), 
            requires_grad=self.learnable_u[adapter_name]
        )
        
        # Pre-multiply sqrt(S) into projection matrices
        self.svft_down_proj[adapter_name] = (Vh.T * sqrt_S).clone().detach().contiguous()  # V.T @ sqrt(S): [d_in, r]
        self.svft_up_proj[adapter_name] = (U * sqrt_S).clone().detach().contiguous()  # U @ sqrt(S): [d_out, r]
        
        self.svft_s0[adapter_name] = S.clone().detach().contiguous()  # [r]
        self.svft_dS[adapter_name] = nn.Parameter(
            torch.zeros(r, device=device), 
            requires_grad=True
        )
        # nn.init.normal_(self.svft_dS[adapter_name], mean=0.0, std=0.01)
        nn.init.uniform_(self.svft_dS[adapter_name], a=1e-5, b=1e-3)
        

    def get_delta(self, x, adapter: str) -> torch.Tensor:
        """
        Compute adapter delta with ΔU parameterization and PiSSA-style sqrt(S) splitting.
        
        Forward pass: x @ down_proj @ (sqrt(S0) + delta) @ up_proj.T
        where down_proj = V.T @ sqrt(S0), up_proj = U @ sqrt(S0)
        
        This operates in variance-weighted space for better gradient flow.
        U_effective = U_base + U_delta (weight decay on U_delta pulls toward U_base)
        """

        # Get pre-scaled projection matrices
        down_proj = self.svft_down_proj[adapter]  # V.T @ sqrt(S0): [d_in, r]
        up_proj_base = self.svft_up_proj[adapter]  # U @ sqrt(S0): [d_out, r]
        
        # Apply learnable U delta in sqrt(S)-scaled space
        if self.learnable_u[adapter]:
            U_delta = self.svft_u_delta[adapter]  # [d_out, r]
            sqrt_S0 = self.svft_s0[adapter].sqrt()  # [r]
            up_proj = up_proj_base + U_delta * sqrt_S0  # Scale delta by sqrt(S0)
        else:
            up_proj = up_proj_base
        
        S0 = self.svft_s0[adapter]  # [r]
        dS = self.svft_dS[adapter]  # [r]
        C = self.svft_coeff[adapter]
        
        # Compute singular value delta in original S space
        scale = S0
        s_eff = C * dS * scale
        
        def soft_clamp(x, n=1.):
            return torch.tanh(x/n)*n
        
        # Bound magnitude
        max_magnitude = 10.0 * S0 + 0.2
        s_eff_diag = soft_clamp(s_eff, max_magnitude)
        
        # Compute ratio: S_total / S0
        # This is the scaling beyond the baked-in sqrt(S0)
        S_total = (S0 + s_eff_diag)
        # Clamp S0 to avoid division by near-zero (especially for tail components)
        sqrt_ratio = S_total / torch.clamp(S0, min=1e-6)
        
        # Sign handling for negative S_total
        # sign_S = torch.sign(S0 + s_eff_diag)  # [r]
        
        # Forward: x @ down_proj @ diag(sqrt_ratio) @ diag(sign) @ up_proj.T
        # down_proj already has sqrt(S0) baked in, we just scale by the ratio
        x_down = x @ down_proj  # [b, s, r] - already sqrt(S0) weighted
        x_scaled = x_down * sqrt_ratio.unsqueeze(0).unsqueeze(0) #* sign_S.unsqueeze(0).unsqueeze(0)  # [b, s, r]
        h = x_scaled @ up_proj.T  # [b, s, d_out]
        
        return h

    def forward(self, x: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        previous_dtype = x.dtype
        
        # Use injected cache from Coconut.recursion_context() if available
        assert len(self.active_adapters) <= 1, "TRM SVFT currently supports only one active adapter at a time."

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            # Check mode from first active adapter
            adapter = self.active_adapters[0]
            mode = self.svft_mode[adapter]
            
            if mode.startswith("replace_"):
                # Replacement mode - compute x @ (U @ S_eff @ V.T).T directly
                # This replaces the base layer output entirely (like original SVFT)
                result = None
                for adapter in self.active_adapters:
                    if adapter not in self.svft_down_proj:
                        continue

                    h = self.get_delta(x, adapter)
                    
                    if result is None:
                        result = h
                    else:
                        result += h  # Multiple adapters (unlikely)
                
                if result is None:
                    result = self.base_layer(x, *args, **kwargs)
            else:
                # Adapter mode - add delta to base layer output
                base_out = self.base_layer(x, *args, **kwargs)
                add_out = torch.zeros_like(base_out)

                for adapter in self.active_adapters:
                    if adapter not in self.svft_down_proj:
                        continue

                    h = self.get_delta(x, adapter)
                    add_out += h

                result = base_out + add_out.to(base_out.dtype)

        result = result.to(previous_dtype)
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for TRM SVFT yet")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerge not implemented for TRM SVFT yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmsvft." + rep


class TRMSvftLinear(nn.Module, TRMSvftLayer):
    """TRM SVFT implemented in a dense layer"""
    
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        **kwargs,
    ) -> None:
        super().__init__()
        TRMSvftLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, **kwargs)

    def forward(self, hidden_states: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        """Forward pass - delegates to TRMSvftLayer.forward"""
        return TRMSvftLayer.forward(self, hidden_states, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmsvft." + rep


class TRMSvftModel(BaseTuner):
    """
    TRM SVFT Model - handles adapter injection into base model.
    Inherits from BaseTuner to integrate with PEFT infrastructure.
    """
    prefix: str = "svft_"
    tuner_layer_cls = TRMSvftLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


    def _create_and_replace(
        self,
        svft_config: TRMSvftAConfig,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key 
        kwargs = {
            "r": svft_config.r,
            "task_type": svft_config.task_type,
            "target_modules": svft_config.target_modules,
            "tail_rank": svft_config.tail_rank,
            "fill_orthonormal": svft_config.fill_orthonormal,
            "learnable_u": svft_config.learnable_u,
            "svft_mode": svft_config.svft_mode,
            "svft_coeff": svft_config.svft_coeff,
            **optional_kwargs,
        }

        if isinstance(target, TRMSvftLinear):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
    
    @staticmethod
    def _create_new_module(adapter_name, target, **kwargs):
        """Create TRMSvftLinear for Linear layers."""
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = TRMSvftLinear(
                target, 
                adapter_name, 
                **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported for TRM SVFT. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module
