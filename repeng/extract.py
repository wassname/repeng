import dataclasses
import os
import typing
import warnings
from typing import Literal, OrderedDict
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
from .svd_steering import svd_steering
from .fisher_steering import natural_gradient_steering
from .losses import compute_reprpo_nll_margin_loss, compute_simpo_loss, compute_reprpo_loss, compute_cosine_loss, compute_kpo_loss



@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[str, np.ndarray]

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
                    - "svd_gradient": SVD on gradients from DPO loss
        Returns:
            ControlVector: The trained vector.
        """
        # the order is [positive, negative, positive, negative, ...]
        train_strs = [s for ex in dataset for s in (ex.positive, ex.negative)]

        # gather hidden states
        act, logprobs, grads = _collect_activations_grads(model, tokenizer, train_strs, hidden_layers, batch_size)

        # compute directions
        dirs = read_representations(
            act, logprobs, grads,
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
        weights_flat = weights.flatten()
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


def read_representations(
    act: dict[str, Tensor],
    logprobs: Tensor,
    grads: dict[str, Tensor | None],
    method: typing.Literal["pca_diff", "pca_center", "umap", "pca_diff_weighted", "pca_center_weighted", "svd_gradient", "fisher_steer"] = "pca_diff",
) -> dict[str, np.ndarray]:
    
    hidden_layers= list(act.keys())
    len_inputs = logprobs.shape[0]//2

    # B. Compute directions
    directions: OrderedDict[str, torch.Tensor] = OrderedDict()
    for layer in tqdm.tqdm(hidden_layers):
        h = act[layer].clone()

        use_empirical_fim = True
        if '_cov' in method:
            use_empirical_fim = False
        
        grad_matrix = grads[layer].clone()
        
        if method == "svd_gradient":
            # For concept extraction, flip negative gradients
            # grad_matrix[1::2] *= -1  # Now all gradients point "toward honesty" 
            directions[layer] = svd_steering(grad_matrix)

            # TODO importance sampling from logprobs
        
        elif "fisher_steer" in method:
            low_dim=None

            if '_dual' in method:
                # Separate Fisher for pos/neg
                grad_pos = grad_matrix[::2]   # [n/2, dim]
                grad_neg = grad_matrix[1::2]  # [n/2, dim]
                
                # Two natural gradients
                low_dim = None
                v_pos = natural_gradient_steering(grad_pos, low_dim=low_dim)
                v_neg = natural_gradient_steering(grad_neg, low_dim=low_dim)

                if '_pos' in method:
                    directions[layer] = v_pos
                elif '_neg' in method:
                    directions[layer] = v_neg
                elif '_diff' in method:
                    # Difference: pos - neg
                    directions[layer] = v_pos - v_neg
                else:
                    # Combine: pos direction and neg direction
                    directions[layer] = (v_pos + v_neg) / 2.0
            elif '_reg0' in method:
                directions[layer] = natural_gradient_steering(
                    grad_matrix, 
                    low_dim=low_dim,
                    lambda_reg=1e-0,
                    use_empirical_fim=use_empirical_fim,
                )
            elif '_reg1' in method:
                directions[layer] = natural_gradient_steering(
                    grad_matrix, 
                    low_dim=low_dim,
                    lambda_reg=1e-1,
                    use_empirical_fim=use_empirical_fim,
                )
            elif '_reg2' in method:
                directions[layer] = natural_gradient_steering(
                    grad_matrix, 
                    low_dim=low_dim,
                    lambda_reg=1e-2,
                    use_empirical_fim=use_empirical_fim,
                )
            elif '_reg3' in method:
                directions[layer] = natural_gradient_steering(
                    grad_matrix, 
                    low_dim=low_dim,
                    lambda_reg=1e-3,
                    use_empirical_fim=use_empirical_fim,
                )
            elif '_reg4' in method:
                directions[layer] = natural_gradient_steering(
                    grad_matrix, 
                    low_dim=low_dim,
                    lambda_reg=1e-4,
                    use_empirical_fim=use_empirical_fim,
                )
            elif '_reg5' in method:
                directions[layer] = natural_gradient_steering(
                    grad_matrix, 
                    low_dim=low_dim,
                    lambda_reg=1e-5,
                    use_empirical_fim=use_empirical_fim,
                )
            else:
                directions[layer] = natural_gradient_steering(
                    grad_matrix, 
                    low_dim=low_dim,
                    use_empirical_fim=use_empirical_fim,
                )

            # make sure the direction has positive personas as +ve (otherwise flip)
            # calculate sign
            projected_hiddens = project_onto_direction(grad_matrix, directions[layer])

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
                directions[layer] *= -1

        else: # PCA-based methods
            # run PCA on difference vectors between positive and negative examples
            train = h[::2] - h[1::2]
            directions[layer] = PCAWeighted(train)

            # make sure the direction has positive personas as +ve (otherwise flip)
            # calculate sign
            projected_hiddens = project_onto_direction(h, directions[layer])

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
                directions[layer] *= -1

    return directions


def _collect_activations_grads(
    model,
    tokenizer,
    inputs: list[str],
    layers_to_edit: list[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray | None]]:
    """Get hidden states and their gradients."""
    assert batch_size % 2 == 0, "batch_size must be even for pos/neg pairs"
    batched_inputs = [inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)]
    if isinstance(model, ControlModel):
        model = model.model
    
    hidden_states: dict[str, list[np.ndarray]] = {layer: [] for layer in layers_to_edit}
    gradients: dict[str, list[np.ndarray]] = {layer: [] for layer in layers_to_edit}
    completion_lprob: list[np.ndarray] = []

    for batch in tqdm.tqdm(batched_inputs, desc="Getting hiddens"):
        encoded_batch = tokenizer(batch, padding=True, return_tensors="pt", padding_side="left").to(model.device)
        attention_mask = encoded_batch["attention_mask"]

        # We don't need gradients wrt model parameters, so turn them off.
        for param in model.parameters():
            param.requires_grad = False

        # But we need to enable them for the embedding layer to build the graph.
        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight.requires_grad = True


        # We need to enable gradients for the DPO loss calculation.
        with torch.enable_grad():
            model.zero_grad()
            with TraceDict(
                model,
                layers=layers_to_edit,
                retain_output=True,
                retain_grad=True,
                detach=False,
            ) as ret:
                outputs = model(**encoded_batch, output_hidden_states=True)

                # We must explicitly tell PyTorch to retain gradients for the non-leaf hidden state tensors.
                for layer in layers_to_edit:
                    ret[layer].output.retain_grad()

                # --- DPO Loss Calculation ---
                lprobs = outputs.logits[:, :-1].log_softmax(-1)
                labels = encoded_batch["input_ids"][:, 1:, None]
                lprobs_for_inputs = torch.gather(input=lprobs, dim=-1, index=labels).squeeze(-1)
                
                label_mask = attention_mask[:, 1:]
                avg_logp_completion = (lprobs_for_inputs * label_mask).sum(-1) / label_mask.sum(-1)

                logp_pos, logp_neg = avg_logp_completion[::2], avg_logp_completion[1::2]
                # loss = compute_simpo_loss(logp_pos, logp_neg)


                hs = outputs.hidden_states[-3] # get layer N-2, this is peak supressed neurons

                # layer = layers_to_edit[-1] # get the last layer we are recording
                # hs = ret[layer].output
                hs_neg = hs[1::2, -1]
                hs_pos = hs[::2, -1]
                loss = compute_reprpo_nll_margin_loss(hs_pos=hs_pos, hs_neg=hs_neg, logp_pos=logp_pos, logp_neg=logp_neg)
                loss.backward()

                # get last non-padded token
                seq_len = label_mask.shape[1]
                last_valid_idx = seq_len - label_mask.flip(-1).to(torch.int64).cpu().argmax(dim=-1) - 1

                # collect activation, grad, avg_logp_completion for each example in batch
                for layer in layers_to_edit:

                    grad = ret[layer].output.grad.detach().float().cpu()
                    last_grad = grad[range(len(last_valid_idx)), last_valid_idx]

                    # combined grads for pos/neg pairs since they share a loss
                    # grad = grad.reshape(len(batch)//2, 2, grad.shape[-1]).sum(dim=1)
                    gradients[layer].append(last_grad)

                    hs = ret[layer].output.detach().float().cpu()
                    last_hs = hs[range(len(last_valid_idx)), last_valid_idx]
                    hidden_states[layer].append(last_hs)

                completion_lprob.extend(avg_logp_completion.detach().cpu().float())



        del outputs, loss, lprobs, lprobs_for_inputs, avg_logp_completion, ret, grad, hs
        model.zero_grad()
        torch.cuda.empty_cache()

    # Restore requires_grad for model parameters
    for param in model.parameters():
        param.requires_grad = True

    # stack layers
    final_grads = {k: torch.vstack(v) if v else None for k, v in gradients.items()}
    hidden_states = {k: torch.vstack(v) for k, v in hidden_states.items()}
    completion_lprob = torch.tensor(completion_lprob)
    return (
        hidden_states, # [batch, hidden_dim]
        completion_lprob, # [batch]
        final_grads, # [batch, hidden_dim] (same for pos/neg pairs since they share a loss)
    )


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = torch.linalg.norm(direction)
    assert not torch.isinf(mag)
    return (H @ direction) / mag
