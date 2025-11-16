import dataclasses
import os
import typing
import warnings
from typing import Callable, Literal, OrderedDict
import gguf
import numpy as np
from sklearn.decomposition import PCA
import torch
from jaxtyping import Float, Int
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm
from baukit import TraceDict

from .control import ControlModel, model_layer_list
from .dataset import DatasetEntry



@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[str, torch.Tensor]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        hidden_layers: typing.Iterable[str] | None = None,
        batch_size: int = 32,
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Options:
                    - "pca_diff": PCA on difference vectors (default)
                    - "pca_center": PCA on centered vectors
                    - "pca_diff_weighted": PCA with weighted diff vectors
                    - "pca_center_weighted": PCA with weighted centered vectors
                    - "umap": UMAP dimensionality reduction
        Returns:
            ControlVector: The trained vector.
        """
        # the order is [positive, negative, positive, negative, ...]
        train_strs = [s for ex in dataset for s in (ex.positive, ex.negative)]

        # gather hidden states
        act, logprobs = _collect_activations(
            model, tokenizer, train_strs, hidden_layers, batch_size
        )

        # compute directions
        dirs = read_representations(
            act, logprobs,
            **kwargs,
        )

        # init class
        return cls(model_type=model.config.model_type, directions=dirs)

    # @classmethod
    # def train_with_sae(
    #     cls,
    #     model: "PreTrainedModel | ControlModel",
    #     tokenizer: PreTrainedTokenizerBase,
    #     sae,
    #     dataset: list[DatasetEntry],
    #     *,
    #     decode: bool = True,
    #     method: typing.Literal["pca_diff", "pca_center", "fisher_steer", "svd_gradient"] = "pca_center",
    #     **kwargs,
    # ) -> "ControlVector":
    #     """
    #     Like ControlVector.train, but using an SAE. It's better! WIP.


    #     Args:
    #         model (PreTrainedModel | ControlModel): The model to train against.
    #         tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
    #         sae (saes.Sae): See the `saes` module for how to load this.
    #         dataset (list[DatasetEntry]): The dataset used for training.
    #         **kwargs: Additional keyword arguments.
    #             decode (bool, optional): Whether to decode the vector to make it immediately usable.
    #                 If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
    #                 to use it. Defaults to True.
    #             max_batch_size (int, optional): The maximum batch size for training.
    #                 Defaults to 32. Try reducing this if you're running out of memory.
    #             method (str, optional): The training method to use. Can be either
    #                 "pca_diff" or "pca_center". Defaults to "pca_center"! This is different
    #                 than ControlVector.train, which defaults to "pca_diff".

    #     Returns:
    #         ControlVector: The trained vector.
    #     """

    #     def transform_hiddens(hiddens: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    #         sae_hiddens = {}
    #         for k, v in tqdm.tqdm(hiddens.items(), desc="sae encoding"):
    #             sae_hiddens[k] = sae.layers[k].encode(v)
    #         return sae_hiddens

    #     # with torch.inference_mode():
    #     dirs = read_representations(
    #         model,
    #         tokenizer,
    #         dataset,
    #         transform_hiddens=transform_hiddens,
    #         method=method,
    #         **kwargs,
    #     )

    #     final_dirs = {}
    #     if decode:
    #         for k, v in tqdm.tqdm(dirs.items(), desc="sae decoding"):
    #             final_dirs[k] = sae.layers[k].decode(v)
    #     else:
    #         final_dirs = dirs

    #     return cls(model_type=model.config.model_type, directions=final_dirs)

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained ControlVector to a llama.cpp .gguf file.
        Note: This file can't be used with llama.cpp yet. WIP!

            vector = ControlVector.train(...)
            vector.export_gguf("path/to/write/vector.gguf")
        """

        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "ControlVector":
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            layer = tensor.name[len("direction.") :]
            if not layer:
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data
        return cls(model_type=model_hint, directions=directions)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[str, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __eq__(self, other: "ControlVector") -> bool:
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[str, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.number) -> "ControlVector":
        directions: dict[str, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(1 / other)
    

def PCAWeighted(train, weights=None):
    """
    https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis\
    """
    if weights is not None:
        weights_flat = weights.flatten().clone()
        # Normalize weights to sum to 1
        weights_norm = weights_flat / weights_flat.sum()
    else:
        weights_norm = torch.ones(train.shape[0])
    
    # Weighted mean and centering
    weighted_mean = torch.sum(train * weights_norm.unsqueeze(-1), dim=0) / weights_norm.sum()
    train = train - weighted_mean
    
    # Apply sqrt of weights
    train_weighted_torch = train * torch.sqrt(weights_norm).unsqueeze(-1)
    
    # Torch SVD (full_matrices=False for efficiency)
    U, S, Vt = torch.linalg.svd(train_weighted_torch, full_matrices=False)
    
    # First PC direction
    direction = Vt[0].cpu()
    return direction


class ComputeHiddens(typing.Protocol):
    def __call__(
        self,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        train_strs: list[str],
        hidden_layers: list[str],
        batch_size: int,
    ) -> dict[str, np.ndarray]: ...



def choose_sign_from_hiddens(direction: torch.Tensor, hiddens: torch.Tensor) -> torch.Tensor:
    projected_hiddens = project_onto_direction(hiddens, direction)
    len_inputs = projected_hiddens.shape[0]//2

    # order is [positive, negative, positive, negative, ...]
    positive_smaller_mean = torch.tensor(
        [
            projected_hiddens[i] < projected_hiddens[i + 1]
            for i in range(0, len_inputs * 2, 2)
        ]
    ).float().mean()
    positive_larger_mean = torch.tensor(
        [
            projected_hiddens[i] > projected_hiddens[i + 1]
            for i in range(0, len_inputs * 2, 2)
        ]
    ).float().mean()

    if positive_smaller_mean > positive_larger_mean:  # type: ignore
        return -direction
    return direction

def read_representations(
    act: dict[str, Tensor],
    logprobs: Tensor,
    method: typing.Literal["pca_diff", "pca_center", "umap", "pca_diff_weighted", "pca_center_weighted"] = "pca_diff",
) -> dict[str, np.ndarray]:
    
    hidden_layers= list(act.keys())

    # B. Compute directions
    directions: OrderedDict[str, torch.Tensor] = OrderedDict()
    for layer in tqdm.tqdm(hidden_layers):
        h = act[layer].clone()

        # PCA-based methods only
        if 'weighted' in method:
            # Normalize logprobs to [0, 2]: Higher prob = higher weight (focus on coherent samples)
            # For pairs, use difference or average; here, weight by pos logprob (more honest = higher weight) 
            pair_probs = logprobs.view(-1, 2).mean(1)  # Average logprob per pair [n_pairs]
            weights = torch.softmax(pair_probs, dim=0) * 2.0  # [0, 2] range
            weights = torch.clamp(weights, min=0.0)  # Non-negative
            # Reshape to match train (interleave for pos/neg, but since train is diff, use pair weights)
            # train will be defined below; just repeat to match pairwise diffs length
            # weights used with train = h[::2] - h[1::2], so shapes match (n_pairs,)
        else:
            weights = None
        # run PCA on difference vectors between positive and negative examples
        train = h[::2] - h[1::2]
        directions[layer] = PCAWeighted(train, weights=weights)

        # make sure the direction has positive personas as +ve (otherwise flip)
        directions[layer] = choose_sign_from_hiddens(directions[layer], h)

    return directions


def _collect_activations(
    model,
    tokenizer,
    inputs: list[str],
    layers_to_edit: list[str],
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Lightweight collection of hidden states and logprobs without gradients.
    Used for PCA-based methods that don't need gradient information.
    
    Returns:
        hidden_states: {layer: [batch, hidden_dim]}
        completion_lprob: [batch]
    """
    assert batch_size % 2 == 0, "batch_size must be even for pos/neg pairs"
    batched_inputs = [inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)]
    if isinstance(model, ControlModel):
        model = model.model
    
    hidden_states: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers_to_edit}
    completion_lprob: list[torch.Tensor] = []
    
    model.eval()
    
    for bi, batch in enumerate(tqdm.tqdm(batched_inputs, desc=f"Getting act for modules={len(layers_to_edit)}")):
        encoded_batch = tokenizer(batch, padding=True, return_tensors="pt", padding_side="left").to(model.device)
        attention_mask = encoded_batch["attention_mask"]
        
        if bi % 10 == 0:
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            with TraceDict(
                model,
                layers=layers_to_edit,
                retain_output=True,
            ) as ret:
                outputs = model(**encoded_batch, output_hidden_states=True)
                
                # Compute logprobs
                lprobs = outputs.logits[:, :-1].log_softmax(-1)
                labels = encoded_batch["input_ids"][:, 1:, None]
                lprobs_for_inputs = torch.gather(input=lprobs, dim=-1, index=labels).squeeze(-1)
                
                label_mask = attention_mask[:, 1:]
                avg_logp_completion = (lprobs_for_inputs * label_mask).sum(-1) / label_mask.sum(-1)
                
                # Get last non-padded token index
                # attention_mask is [batch, seq_len] with 1s for real tokens, 0s for padding
                # For left padding: [0,0,0,1,1,1] -> last index is seq_len-1
                # For right padding: [1,1,1,0,0,0] -> need to find last 1
                # Flip and argmax finds first 1 from right, subtract from end
                last_valid_idx = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[-1]).argmax(dim=-1).cpu()
                
                # Collect activations from each layer
                for layer in layers_to_edit:
                    hs = ret[layer].output.detach().float().cpu()
                    last_hs = hs[range(len(last_valid_idx)), last_valid_idx]
                    hidden_states[layer].append(last_hs)
                
                completion_lprob.append(avg_logp_completion.detach().cpu().float())
        
        del outputs, lprobs, lprobs_for_inputs, ret
        torch.cuda.empty_cache()
    
    # Stack results
    hidden_states = {k: torch.vstack(v) for k, v in hidden_states.items()}
    completion_lprob = torch.cat(completion_lprob)
    
    return hidden_states, completion_lprob


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = torch.linalg.norm(direction)
    assert not torch.isinf(mag)
    return (H @ direction) / mag



