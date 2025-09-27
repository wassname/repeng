import dataclasses
import functools
import re
import typing
from typing import Dict, List, Optional, Iterable, Tuple, Union, Callable, Any, TYPE_CHECKING
from jaxtyping import Float
import warnings
from collections import OrderedDict
from baukit import TraceDict
import torch
from torch import Tensor, nn

import contextlib
from transformers import PretrainedConfig, PreTrainedModel

if TYPE_CHECKING:
    from .extract import ControlVector


def noop_edit(output, layer, inputs):
    return output



@dataclasses.dataclass
class BlockControlParams:
    control: Tensor | None = None
    normalize: bool = False
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()

class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(self, model: PreTrainedModel, directions: OrderedDict[str, BlockControlParams] = {}) -> None:
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
        """

        super().__init__()
        self.model = model
        if isinstance(model, ControlModel):
            warnings.warn(
                "Trying to wrap a wrapped model! Probably not what you want! Try calling .unwrap first."
            )
            model = model.model

        self.directions = directions
        self.reset()

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device
    
    @property
    def dtype(self) -> torch.dtype:
        p = next(iter(self.model.parameters()))
        return p.dtype

    def unwrap(self) -> PreTrainedModel:
        return self.model

    def set_control(
        self, control: "ControlVector", coeff: float = 1.0, **kwargs
    ) -> None:
        """
        Set a `ControlVector` for the layers this ControlModel handles, with a strength given
        by `coeff`. (Negative `coeff` values invert the control vector, e.g. happiness→sadness.)
        `coeff` defaults to `1.0`.

        Additional kwargs:
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """
        
        if control is None:
            return self.reset()
        else:
            self.directions = control.directions
            self.edit_fn = functools.partial(
                baukit_dir_add_hook, directions=control.directions, coeff=coeff
            )

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.edit_fn = noop_edit

    def _steer(self, fn: Callable, *args, **kwargs):
        with TraceDict(
            self.model, 
            layers=list(self.directions.keys()),
            retain_output=False,
            detach=True,
            edit_output=self.edit_fn,
        ) as td:
            return fn(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._steer(self.model.forward, *args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._steer(self.model.generate, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._steer(self.model.__call__, *args, **kwargs)



def model_layer_list(model: ControlModel | PreTrainedModel) -> torch.nn.ModuleList:
    if isinstance(model, ControlModel):
        model = model.model

    target_suffixes = [
        "repeng_layers",  # override
        "model.layers",  # llama, mistral, gemma, qwen, ...
        "transformer.h",  # gpt-2
    ]
    for suffix in target_suffixes:
        candidates = [
            v
            for k, v in model.named_modules()
            if k.endswith(suffix) and isinstance(v, torch.nn.ModuleList)
        ]
        if len(candidates) == 1:
            return candidates[0]

    raise ValueError(
        f"don't know how to get layer list for {type(model)}! try assigning `model.repeng_layers = ...` to override this search."
    )


def get_available_layers(model, regex_filter: Optional[str] = None, layer_range: Optional[Tuple[int, int]] = None) -> Tuple[List[str], List[str]]:
    """Find available layers in a model using named_parameters style paths

    Usage:
        ```
        # all blocks and layers with weights
        get_available_layers(model, layer_range=(0.1, 0.9))
        # get hidden states from layer 10% to 90%
        get_available_layers(model, regex_filter="\d+$", layer_range=(0.1, 0.9))
        # ['model.layers.10', 'model.layers.11',...]
        # get k projections from layer 10 to 20
        get_available_layers(model, regex_filter="k_proj$", layer_range=(10, 20))
        ```

    Outputs:
        - short names with layer numbers replaced by {N}, e.g. `['model.layers.{N}.k_proj', ...]`
        - full names with layer numbers, e.g. `['model.layers.10.k_proj', 'model.layers.11',...]`
    
    """

    # linear layers
    available_layers = [k.replace(".weight", "") for k, v in model.named_parameters()]

    # parents/blocks
    for l in available_layers:
        while len(l) > 0:
            l = ".".join(l.split(".")[:-1])
            if l not in available_layers and l != "":
                available_layers.append(l)

    # filter by range
    n_layers = len(model_layer_list(model))
    if layer_range is not None:
        # handle fractions
        if all(isinstance(x, float) for x in layer_range):
            layer_range = (int(layer_range[0] * n_layers), int(layer_range[1] * n_layers))

        # handle negative
        for i, n in enumerate(layer_range):
            if n < 0:
                layer_range[i] = n_layers + n

        # filter to range
        layer_range = list(range(*layer_range))
        available_layers = [
            s for s in available_layers if any(f".{i}." in s or s.endswith(f".{i}") for i in layer_range)
        ]

    if regex_filter is not None:
        available_layers = [s for s in available_layers if re.search(regex_filter, s)]

    # remove layer numbers
    short_available_layers = sorted(
        set(re.sub(r"\d+", "{N}", s) for s in available_layers)
    )
    return short_available_layers, available_layers


@torch.no_grad()
def baukit_dir_add_hook(
    output: Float[Tensor, "... d_act"],
    layer: str,
    inputs,
    directions: Dict[str, Float[Tensor, "... d_act"]],
    coeff: float = 1.0,
):
    """
    edit layer output by adding, used in baukit
    """
    if isinstance(output, tuple):
        modified = output[0]
    else:
        modified = output
    direction = directions[layer].to(device=output.device, dtype=output.dtype) * coeff
    modified = modified + direction
    if isinstance(output, tuple):
        output = (modified,) + output[1:]
    else:
        output = modified
    return output



@contextlib.contextmanager
def steer(model: 'PreTrainedModel', vector: "ControlVector", coeff: float):
    """
    Usage:
        with steer(model, vector, coeff):
            out = model.generate()
    """
    if isinstance(model, ControlModel):
        model = model.model
    layers=list(vector.directions.keys())
    if coeff==0:
        edit_fn = noop_edit
    else:
        edit_fn = functools.partial(
            baukit_dir_add_hook, directions=vector.directions, coeff=coeff
        )
    with TraceDict(
        model, 
        layers=layers,
        retain_output=False,
        detach=True,
        edit_output=edit_fn,
    ) as td:
        yield model

