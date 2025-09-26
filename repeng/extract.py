import dataclasses
import os
import typing
import warnings
from typing import Literal
import gguf
import numpy as np
from sklearn.decomposition import PCA
import torch
from jaxtyping import Float, Int
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm
import baukit.nethook as nethook

from .control import ControlModel, model_layer_list

from .svd_steering import svd_steering
try:
    from .fisher_steering import fisher_steering, get_fisher_matrices
except ImportError:
    fisher_steering = None
    get_fisher_matrices = None


@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str


@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
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
                compute_hiddens (Callable, optional): Override hidden state computation.
                    See signature of `read_representations`.
                transform_hiddens (Callable, optional): Transform the hidden states after
                    they are computed. See signature of `read_representations`.
                low_dim (int, optional): Low dimension for projection in SVD. Default 512.
                rank (int, optional): Number of SVD components. Default 2.

        Returns:
            ControlVector: The trained vector.
        """
        dirs = read_representations(
            model,
            tokenizer,
            dataset,
            **kwargs,
        )
        return cls(model_type=model.config.model_type, directions=dirs)

    @classmethod
    def train_with_sae(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        sae,
        dataset: list[DatasetEntry],
        *,
        decode: bool = True,
        method: typing.Literal["pca_diff", "pca_center", "fisher_steer", "svd_gradient"] = "pca_center",
        **kwargs,
    ) -> "ControlVector":
        """
        Like ControlVector.train, but using an SAE. It's better! WIP.


        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            sae (saes.Sae): See the `saes` module for how to load this.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                decode (bool, optional): Whether to decode the vector to make it immediately usable.
                    If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
                    to use it. Defaults to True.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_center"! This is different
                    than ControlVector.train, which defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """

        def transform_hiddens(hiddens: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            sae_hiddens = {}
            for k, v in tqdm.tqdm(hiddens.items(), desc="sae encoding"):
                sae_hiddens[k] = sae.layers[k].encode(v)
            return sae_hiddens

        # with torch.inference_mode():
        dirs = read_representations(
            model,
            tokenizer,
            dataset,
            transform_hiddens=transform_hiddens,
            method=method,
            **kwargs,
        )

        final_dirs = {}
        if decode:
            for k, v in tqdm.tqdm(dirs.items(), desc="sae decoding"):
                final_dirs[k] = sae.layers[k].decode(v)
        else:
            final_dirs = dirs

        return cls(model_type=model.config.model_type, directions=final_dirs)

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
            try:
                layer = int(tensor.name.split(".")[1])
            except (IndexError, ValueError):
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
        directions: dict[int, np.ndarray] = {}
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
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.number) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(1 / other)
    

def PCAWeighted(train, weights=None, device="cpu"):
    """
    https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis\
    """
    if weights is not None:
        weights_flat = weights.flatten()
        # Normalize weights to sum to 1
        weights_norm = weights_flat / weights_flat.sum()
    else:
        weights_norm = np.ones(train.shape[0])
    
    # Weighted mean and centering
    weighted_mean = np.average(train, axis=0, weights=weights_norm)
    train = train - weighted_mean

    # Convert to torch tensor (move to GPU if available for speed)
    train_torch = torch.from_numpy(train).to(device, dtype=torch.float32)
    weights_norm_torch = torch.from_numpy(weights_norm).to(device, dtype=torch.float32)
    
    # Apply sqrt of weights
    train_weighted_torch = train_torch * torch.sqrt(weights_norm_torch).unsqueeze(-1)
    
    # Torch SVD (full_matrices=False for efficiency)
    U, S, Vt = torch.linalg.svd(train_weighted_torch, full_matrices=False)
    
    # First PC direction (move back to CPU/NumPy for compatibility)
    direction = Vt[0].cpu().numpy().astype(np.float32)
    return direction


class ComputeHiddens(typing.Protocol):
    def __call__(
        self,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        train_strs: list[str],
        hidden_layers: list[int],
        batch_size: int,
    ) -> dict[int, np.ndarray]: ...


def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs=None,
    reference_rejected_logprobs=None,
    β=0.1,
):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        β: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as β -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """
    # TODO add reference free DPO (was it KPO, or rainbowPO? I think that just used the ratio of cross entropies?) entropy normalised https://github.com/CapitalOne-Research/RainbowPO/blob/main/trl/trainer/dpo_trainer.py#L1148
    # TODO add the thing to avoid degenerate solutions where it just makes the rejected very low but doens't make the chosen big
    # TODO mean of logprobs like ipo to avoid length bias

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    logits = model_logratios
    if reference_chosen_logprobs is not None and reference_rejected_logprobs is not None:
        reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
        logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(β * logits)

    # .mean() to average over the samples in the batch
    return losses.mean()


def _collect_activations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list, # Can be list[str] or list[DatasetEntry]
    layers: list[int],
    batch_size: int,
    loss_fn: typing.Callable[[typing.Any, typing.Any], Float[Tensor, ""]] | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Internal helper to collect hidden states and optional gradients for specified layers.
    It is loss-function-agnostic.
    """
    device = model.device
    results = {layer: {'hiddens': [], 'grads': []} for layer in layers}
    
    activations = {}
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0].detach()
            if loss_fn is not None:
                h.requires_grad_(True)
            activations[layer_idx] = h
        return hook_fn

    for i in tqdm.tqdm(range(0, len(inputs), batch_size), desc="Collecting Activations", leave=False):
        batch = inputs[i : i + batch_size]
        
        # Tokenize based on input type
        if isinstance(batch[0], str):
            tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        elif isinstance(batch[0], DatasetEntry):
            pos_prompts = [ex.positive for ex in batch]
            neg_prompts = [ex.negative for ex in batch]
            # HACK: For DPO, we need logprobs for both, so we pass them both through.
            # This assumes the loss_fn knows how to handle the concatenated batch.
            all_prompts = pos_prompts + neg_prompts
            tokenized = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        handles = [layer.register_forward_hook(make_hook(layer)) for i, layer in enumerate(layers)]

        with torch.enable_grad():
            model.zero_grad()
            loss = loss_fn(model, tokenized)
            if loss is not None:
                loss.backward()

        for handle in handles:
            handle.remove()

        sequence_lengths = tokenized.attention_mask.sum(dim=1) - 1
        for layer in layers:
            h_full = activations[layer]
            h_last_token = h_full[torch.arange(h_full.shape[0]), sequence_lengths].cpu().numpy()
            results[layer]['hiddens'].append(h_last_token)
            
            if loss_fn and h_full.grad is not None:
                g_full = h_full.grad
                g_last_token = g_full[torch.arange(g_full.shape[0]), sequence_lengths].cpu().numpy()
                results[layer]['grads'].append(g_last_token)

    for layer in layers:
        results[layer]['hiddens'] = np.concatenate(results[layer]['hiddens'], axis=0)
        if results[layer]['grads']:
            results[layer]['grads'] = np.concatenate(results[layer]['grads'], axis=0)

    return results


def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Literal["pca_diff", "pca_center", "umap", "pca_diff_weighted", "pca_center_weighted", "svd_gradient", "fisher_steer"] = "pca_diff",
    compute_hiddens: ComputeHiddens | None = None,
    transform_hiddens: (
        typing.Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
) -> dict[int, np.ndarray]:
    if hidden_layers is None:
        hidden_layers = model_layer_list(model)
    hidden_layers = list(hidden_layers)

    # A. Collect necessary data based on method
    dpo_grads, F_matrices, pca_hiddens = {}, {}, {}

    if method in ["svd_gradient", "fisher_steer"]:
        def dpo_loss_fn(model, tokenized):
            # Assumes tokenized batch is [pos..., neg...]
            batch_size = tokenized.input_ids.shape[0] // 2
            outputs = model(**tokenized)
            logits = outputs.logits # [2*batch, seq_len, vocab]
            # Get logprobs for the actual completions
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            # HACK: This gathers the logprob of the actual next token
            labels = tokenized.input_ids[:, 1:]
            all_logprobs = logprobs[:, :-1, :].gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            
            pos_logprobs, neg_logprobs = all_logprobs.split(batch_size, dim=0)
            loss = -compute_dpo_loss(pos_logprobs.sum(-1), neg_logprobs.sum(-1))
            return loss.mean()

        dpo_data = _collect_activations(model, tokenizer, inputs, hidden_layers, batch_size, loss_fn=dpo_loss_fn)
        for layer in hidden_layers:
            # The DPO grad is G(pos) - G(neg)
            pos_grads, neg_grads = np.split(dpo_data[layer]['grads'], 2)
            dpo_grads[layer] = pos_grads - neg_grads

    if method == "fisher_steer":
        F_matrices = get_fisher_matrices(model, tokenizer, [ex.positive for ex in inputs], hidden_layers, batch_size)

    if "pca" in method:
        all_prompts = [ex.positive for ex in inputs] + [ex.negative for ex in inputs]
        pca_data = _collect_activations(model, tokenizer, all_prompts, hidden_layers, batch_size, loss_fn=None)
        for layer in hidden_layers:
            pca_hiddens[layer] = pca_data[layer]['hiddens']

    # B. Compute directions
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers):
        if method == "svd_gradient":
            grad_matrix = torch.from_numpy(dpo_grads[layer])
            directions[layer] = svd_steering(grad_matrix)
        
        elif method == "fisher_steer":
            grad_matrix = torch.from_numpy(dpo_grads[layer])
            F_matrix = F_matrices[layer]
            directions[layer] = fisher_steering(grad_matrix, F_matrix)

        else: # PCA-based methods
            h = pca_hiddens[layer]
            # run PCA on difference vectors between positive and negative examples
            train = h[::2] - h[1::2]
            directions[layer] = PCAWeighted(train)

        # make sure the direction has positive personas as +ve (otherwise flip)
        # calculate sign
        projected_hiddens = project_onto_direction(h, directions[layer])

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions

def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[int, np.ndarray | None]]:
    """Get hidden states and their gradients."""
    assert batch_size % 2 == 0, "batch_size must be even for pos/neg pairs"
    batched_inputs = [inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)]
    hidden_states: dict[int, list[np.ndarray]] = {layer: [] for layer in hidden_layers}
    gradients: dict[int, list[np.ndarray]] = {layer: [] for layer in hidden_layers}
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
            # output_hidden_states=True will give us the tensors we need.
            # They will be part of the computation graph.
            outputs = model(**encoded_batch, output_hidden_states=True)

            # We must explicitly tell PyTorch to retain gradients for the non-leaf hidden state tensors.
            for layer in hidden_layers:
                outputs.hidden_states[layer].retain_grad()

            # --- DPO Loss Calculation ---
            lprobs = outputs.logits[:, :-1].log_softmax(-1)
            labels = encoded_batch["input_ids"][:, 1:, None]
            lprobs_for_inputs = torch.gather(input=lprobs, dim=-1, index=labels).squeeze(-1)
            
            label_mask = attention_mask[:, 1:]
            avg_logp_completion = (lprobs_for_inputs * label_mask).sum(-1) / label_mask.sum(-1)
            completion_lprob.extend(avg_logp_completion.detach().cpu().float().numpy())

            logp_pos, logp_neg = avg_logp_completion[::2], avg_logp_completion[1::2]
            loss = compute_dpo_loss(logp_pos, logp_neg)

            # --- Gradient Calculation ---
            # We backpropagate the loss. The gradients will accumulate on the hidden state tensors.
            loss.backward()

            # --- Hidden State and Gradient Extraction ---
            completion_lprob.extend(avg_logp_completion.detach().cpu().float().numpy())
            last_positions = [mask.nonzero(as_tuple=True)[0][-1].item()-1 for mask in attention_mask]
            for layer in hidden_layers:
                hs_tensor = outputs.hidden_states[layer]
                # The gradient is stored on the tensor we called .retain_grad() on.
                grad_tensor = hs_tensor.grad
                if grad_tensor is None:
                    raise ValueError(f"No gradients computed for layer {layer}, cannot use svd_gradient method.")

                for i, pos in enumerate(last_positions):
                    # Extract the hidden state for the last token.
                    hs = hs_tensor[i, pos]
                    hidden_states[layer].append(hs.detach().float().cpu().numpy())
                    
                    # Extract the corresponding gradient from the full gradient tensor.
                    grad = grad_tensor[i, pos]
                    if grad.sum()==0:
                        raise ValueError(f"Zero gradient encountered for layer {layer}, cannot use svd_gradient method.")
                    gradients[layer].append(grad.detach().float().cpu().numpy())

        del outputs, loss, lprobs, lprobs_for_inputs, avg_logp_completion
        model.zero_grad()
        torch.cuda.empty_cache()

    # Restore requires_grad for model parameters
    for param in model.parameters():
        param.requires_grad = True

    # If no gradients were computed at all for a layer, the list will be empty.
    final_grads = {k: np.vstack(v) if v else None for k, v in gradients.items()}

    return (
        {k: np.vstack(v) for k, v in hidden_states.items()},
        np.array(completion_lprob),
        final_grads,
    )


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag
