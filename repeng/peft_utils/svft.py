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
from torch import Tensor, device
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
    Configuration for TRM SVFT adapter with SVDSteering rotations.
    
    SVD-based steering with PiSSA decomposition: W = U @ S @ V^T + W_res
    - Top-r SVD components (U, S, V) for principal directions
    - Residual W_res captures remaining variance
    - SSVD rotations (selective rotation of U/V singular vectors)
    - Learnable singular value scaling (add/mult)
    - OFT block-diagonal structure (parameter efficiency for rotations)
    """
    # SVFT-specific parameters
    r: int = field(default=16, metadata={"help": "SVD rank for principal components"})
    rotate_u: bool = field(
        default=False,
        metadata={"help": "Learn rotation on U singular vectors (SVDSteering-style)"}
    )
    rotate_v: bool = field(
        default=True,
        metadata={"help": "Learn rotation on V singular vectors (SVDSteering-style)"}
    )
    rotation_method: Literal["matrix_exp", "cayley", "block_diagonal"] = field(
        default="cayley",
        metadata={"help": "Rotation parameterization: matrix_exp, cayley, or block_diagonal"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Block size for block_diagonal rotation (must divide r)"}
    )
    scale_s: Literal["add", "mult", "none"] = field(
        default="add",
        metadata={"help": "S scaling mode: add (S + delta_s), mult (lambda_s * S), or none (frozen S)"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Steering coefficient for rotations (1.0 = forward, -1.0 = reverse, 0.0 = disabled)"}
    )
    # steer_s: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to apply steering to singular value scaling"}
    # )
    
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
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class TRMSvftLayer(BaseTunerLayer):
    """
    TRM SVFT layer with SVDSteering-style decomposition.
    
    W = U @ S @ V^T + W_res where:
    - U, V: Top-r singular vectors (can be rotated)
    - S: Top-r singular values (can be scaled via dS)
    - W_res: Residual matrix (frozen)
    """

    adapter_layer_names = ("svft_delta_s", "svft_loglambda_s", "svft_rotation_params_u", "svft_rotation_params_v")
    other_param_names = ("svft_u", "svft_v", "svft_s", "svft_w_res", "svft_scale_s", "svft_alpha", "svft_r", "svft_rotate_u", "svft_rotate_v", "svft_rotation_method", "svft_block_size")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.svft_r = {}
        self.svft_rotate_u = {}
        self.svft_rotate_v = {}
        self.svft_rotation_method = {}
        self.svft_block_size = {}
        self.svft_scale_s = {}
        self.svft_alpha = {}
        # self.svft_steer_s = {}
        
        # SVD components (per adapter) - simplified naming like SVDSteering
        self.svft_u = BufferDict({})  # U: [d_out, r]
        self.svft_v = BufferDict({})  # V: [d_in, r]
        self.svft_s = BufferDict({})  # S: [r]
        self.svft_w_res = BufferDict({})  # W_res: [d_out, d_in]
        
        # Learnable S scaling (DeLoRA-style)
        self.svft_delta_s = nn.ParameterDict({})  # add: S + delta_s
        self.svft_loglambda_s = nn.ParameterDict({})  # mult: lambda_s * S
        
        # Rotation parameters (SVDSteering-style)
        self.svft_rotation_params_u = nn.ParameterDict({})
        self.svft_rotation_params_v = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False

        # Marker for Coconut to find TRM layers
        self._recursion_cache = None

        self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        scale_s,
        alpha,
        r,
        rotate_u,
        rotate_v,
        rotation_method,
        block_size,
        # steer_s,
        **kwargs
    ) -> None:
        """
        Initialize SVFT adapter with simple top-r SVD + residual (PiSSA-style).
        """
        if adapter_name in self.svft_u:
            return  # Already initialized

        self.svft_scale_s[adapter_name] = scale_s
        self.svft_alpha[adapter_name] = float(alpha)
        self.svft_r[adapter_name] = r
        self.svft_rotate_u[adapter_name] = rotate_u
        self.svft_rotate_v[adapter_name] = rotate_v
        self.svft_rotation_method[adapter_name] = rotation_method
        self.svft_block_size[adapter_name] = block_size
        # self.svft_steer_s[adapter_name] = steer_s

        # Get base weight
        base_weight = self.get_base_layer().weight
        
        # Dequantize if needed
        if isinstance(base_weight, Params4bit):
            base_weight = bnb.functional.dequantize_4bit(base_weight.data, base_weight.quant_state)
        elif isinstance(base_weight, Int8Params):
            base_weight = bnb.functional.dequantize_8bit(base_weight.data, base_weight.quant_state)
        
        base_weight = base_weight.float()  # [out, in]
        device = base_weight.device

        # Simple top-r SVD (like SVDSteering snippet)
        U_full, S_full, Vh_full = torch.linalg.svd(base_weight, full_matrices=False)
        
        U = U_full[:, :r]  # [d_out, r]
        S = S_full[:r]     # [r]
        Vh = Vh_full[:r, :]  # [r, d_in]
        V = Vh.T           # [d_in, r]
        
        # Compute residual (PiSSA-style)
        W_principal = U @ torch.diag(S) @ Vh
        W_res = base_weight - W_principal
        
        # Store frozen components
        self.svft_u[adapter_name] = U.clone().detach().contiguous()
        self.svft_v[adapter_name] = V.clone().detach().contiguous()
        self.svft_s[adapter_name] = S.clone().detach().contiguous()
        self.svft_w_res[adapter_name] = W_res.clone().detach().contiguous()
        
        # Learnable S scaling (modified to be reversible DeLoRA/PiSSA-style)
        if scale_s == "add":
            self.svft_delta_s[adapter_name] = nn.Parameter(
                torch.zeros(r, device=device), 
                requires_grad=True
            )
            nn.init.uniform_(self.svft_delta_s[adapter_name], a=1e-5, b=1e-3)
        elif scale_s == "mult":
            self.svft_loglambda_s[adapter_name] = nn.Parameter(
                torch.zeros(r, device=device), 
                requires_grad=True
            )
            nn.init.trunc_normal_(self.svft_loglambda_s[adapter_name], std=0.002)



        def initialize_skew_symmetric_matrix(*args, **kwargs):
            """With contrastive steering coeff=+1 and coeff=-1 produce identical outputs initially, so gradients are zero. Small random init is important for learning as it breaks symmetry."""
            x = torch.zeros(*args, **kwargs)
            # Option B: Draw from skew-symmetric distribution directly
            nn.init.trunc_normal_(x, std=0.002)
            x = x - x.T
            return x
        
        # Initialize rotation parameters (reversible OFT,SSVD-style)
        if rotate_u:
            if rotation_method == "block_diagonal":
                assert block_size is not None and r % block_size == 0, f"block_size {block_size} must divide r {r}"
                num_blocks = r // block_size
                self.svft_rotation_params_u[adapter_name] = nn.Parameter(
                    initialize_skew_symmetric_matrix(num_blocks, block_size, block_size, device=device)
                )
            else:
                self.svft_rotation_params_u[adapter_name] = nn.Parameter(
                    initialize_skew_symmetric_matrix(r, r, device=device)
                )
        
        if rotate_v:
            if rotation_method == "block_diagonal":
                assert block_size is not None and r % block_size == 0, f"block_size {block_size} must divide r {r}"
                num_blocks = r // block_size
                self.svft_rotation_params_v[adapter_name] = nn.Parameter(
                    initialize_skew_symmetric_matrix(num_blocks, block_size, block_size, device=device)
                )
            else:
                self.svft_rotation_params_v[adapter_name] = nn.Parameter(
                    initialize_skew_symmetric_matrix(r, r, device=device)
                )
    def _get_rotation(
        self, 
        params: Float[Tensor, "... r r"],
        alpha: float,
        rotation_method: str,
    ) -> Float[Tensor, "... r r"]:
        """Compute rotation matrix from learnable parameters (SVDSteering-style).
        
        Args:
            params: Rotation parameters (unconstrained)
            alpha: Steering coefficient (1.0 = forward, -1.0 = reverse)
            rotation_method: Rotation parameterization method
        
        Returns:
            Orthogonal rotation matrix R ∈ SO(r)
        """
        if rotation_method == "block_diagonal":
            # params shape: [num_blocks, block_size, block_size]
            blocks = []
            for block_params in params:
                A = block_params - block_params.T  # skew-symmetric
                R_block = self._rotation_from_skew(A, alpha, rotation_method)
                blocks.append(R_block)
            return torch.block_diag(*blocks)
        else:
            # Full rotation: params shape: [r, r]
            A = params - params.T  # skew-symmetric projection
            return self._rotation_from_skew(A, alpha, rotation_method)
    
    def _rotation_from_skew(
        self,
        A: Float[Tensor, "r r"],
        alpha: float,
        rotation_method: str,
    ) -> Float[Tensor, "r r"]:
        """Compute rotation from skew-symmetric matrix."""
        if rotation_method in ["matrix_exp", "block_diagonal"]:
            # Matrix exponential: exp(αA)
            return torch.matrix_exp(alpha * A)
        elif rotation_method == "cayley":
            # Cayley transform: (I - αA/2)^{-1} (I + αA/2)
            # More efficient than matrix_exp, same reversibility
            I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            X = alpha * A / 2
            return torch.linalg.solve(I - X, I + X)
        else:
            raise ValueError(f"Unknown rotation method: {rotation_method}")

    def get_adapted_output(self, x, adapter: str) -> torch.Tensor:
        """
        Compute adapter output (SVDSteering-style).
        
        W_adapted = U_rot @ diag(S_scaled) @ V_rot^T + W_res
        Forward: x @ V_rot @ diag(S_scaled) @ U_rot^T + x @ W_res^T
        
        Note: alpha scales rotations only (steering strength), not S
        """
        alpha = self.svft_alpha[adapter]
        # steer_s = self.svft_steer_s[adapter]
        
        # Get frozen bases
        U = self.svft_u[adapter]  # [d_out, r]
        V = self.svft_v[adapter]  # [d_in, r]
        S = self.svft_s[adapter]  # [r]
        W_res = self.svft_w_res[adapter]  # [d_out, d_in]
        
        # Apply rotations (alpha scales rotation strength, not magnitude)
        if self.svft_rotate_v[adapter] and adapter in self.svft_rotation_params_v:
            R_v = self._get_rotation(
                self.svft_rotation_params_v[adapter], 
                alpha=alpha,
                rotation_method=self.svft_rotation_method[adapter]
            )
            V_rot = V @ R_v  # [d_in, r]
        else:
            V_rot = V
        
        if self.svft_rotate_u[adapter] and adapter in self.svft_rotation_params_u:
            R_u = self._get_rotation(
                self.svft_rotation_params_u[adapter],
                alpha=alpha,
                rotation_method=self.svft_rotation_method[adapter]
            )
            U_rot = U @ R_u  # [d_out, r]
        else:
            U_rot = U
        
        # Scale S independently (no alpha - this controls magnitude, not direction)
        scale_mode = self.svft_scale_s[adapter]
        if scale_mode == "add":
            delta_s = self.svft_delta_s[adapter]  # [r]
            # if steer_s:
            #     delta_s = delta_s * alpha
            # S_scaled = S + delta_s

            # OR
            S_scaled = S + alpha * torch.tanh(delta_s) * S
        elif scale_mode == "mult":
            loglambda_s = self.svft_loglambda_s[adapter]
            S_scaled = (loglambda_s * alpha).exp() * S
        else:  # "none"
            S_scaled = S
        
        # Efficient forward: x @ V_rot @ diag(S_scaled) @ U_rot^T
        x_projected = x @ V_rot  # [..., r]
        x_scaled = x_projected * S_scaled  # [..., r] - broadcast multiply
        x_transformed = x_scaled @ U_rot.T  # [..., d_out]
        
        # Add residual contribution
        x_residual = x @ W_res.T  # [..., d_out]
        
        return x_transformed + x_residual

    def forward(self, x: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        previous_dtype = x.dtype
        
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

            # Always compute full adapted weight (no mode switching)
            result = None
            for adapter in self.active_adapters:
                if adapter not in self.svft_u:
                    continue

                h = self.get_adapted_output(x, adapter)
                
                if result is None:
                    result = h
                else:
                    result += h  # Multiple adapters (unlikely)
            
            if result is None:
                result = self.base_layer(x, *args, **kwargs)

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
            "rotate_u": svft_config.rotate_u,
            "rotate_v": svft_config.rotate_v,
            "rotation_method": svft_config.rotation_method,
            "block_size": svft_config.block_size,
            "scale_s": svft_config.scale_s,
            "alpha": svft_config.alpha,
            # "steer_s": svft_config.steer_s,
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
