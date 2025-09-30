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
from .analyze_vectors.svd_steering import svd_steering
from .analyze_vectors.fisher_steering import natural_gradient_steering
from .train.inner_contrastive_loss import contrastive_steering_loss_noref



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
                    - "svd_gradient": SVD on gradients from DPO loss
        Returns:
            ControlVector: The trained vector.
        """
        # the order is [positive, negative, positive, negative, ...]
        train_strs = [s for ex in dataset for s in (ex.positive, ex.negative)]

        # gather hidden states
        act, logprobs, grads, feat_grad_norms = _collect_activations_grads(
            model, tokenizer, train_strs, hidden_layers, batch_size
        )

        # compute directions
        dirs = read_representations(
            act, logprobs, grads, feat_grad_norms,
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



def _choose_sign_from_grads(direction: torch.Tensor, grad_matrix: torch.Tensor) -> torch.Tensor:
    """
    Fix direction sign using first-order loss change.
    We want +v on positives and -v on negatives to reduce loss:
      mean((g_neg - g_pos) @ v) >= 0
    If the mean is negative, flip v.
    """
    v = direction
    g_pos = grad_matrix[::2]   # [n/2, d]
    g_neg = grad_matrix[1::2]  # [n/2, d]
    score = torch.mean((g_neg - g_pos) @ v)
    if torch.isnan(score):
        return v  # keep as-is if degenerate
    return v if score >= 0 else -v

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
    grads: dict[str, Tensor | None],
    feat_grad_norms: dict[str, Tensor | None] = None,
    method: typing.Literal["pca_diff", "pca_center", "umap", "pca_diff_weighted", "pca_center_weighted", "svd_gradient", "fisher_steer", "hvp_steer"] = "pca_diff",
) -> dict[str, np.ndarray]:
    
    hidden_layers= list(act.keys())

    # B. Compute directions
    directions: OrderedDict[str, torch.Tensor] = OrderedDict()
    for layer in tqdm.tqdm(hidden_layers):
        h = act[layer].clone()

        fim = 'improved_empirical'
        if '_cov' in method:
            fim = 'covariance'
        elif "_emp" in method:
            fim = 'empirical'
            
        grad_matrix = grads[layer].clone()
        dim = grad_matrix.shape[-1]
        
        if method == "svd_gradient":
            # For concept extraction, flip negative gradients
            # grad_matrix[1::2] *= -1  # Now all gradients point "toward honesty" 
            directions[layer] = svd_steering(grad_matrix)

            # TODO importance sampling from logprobs
        
        elif "fisher_steer" in method:
            low_dim=None



            lambda_reg = 1e-2
            if '_reg0' in method:
                lambda_reg = 0.0
            elif '_reg1' in method:
                lambda_reg = 1e-1
            elif '_reg2' in method:
                lambda_reg = 1e-2
            elif '_reg3' in method:
                lambda_reg = 1e-3
            elif '_reg4' in method:
                lambda_reg = 1e-4
            elif '_reg5' in method:
                lambda_reg = 1e-5

            if '_dual' in method:
                # Separate Fisher for pos/neg
                grad_pos = grad_matrix[::2]   # [n/2, dim]
                grad_neg = grad_matrix[1::2]  # [n/2, dim]
                if feat_grad_norms is not None and feat_grad_norms[layer] is not None:
                    norms_pos = feat_grad_norms[layer][::2]
                    norms_neg = feat_grad_norms[layer][1::2]
                else:
                    norms_pos = norms_neg = None
                
                # Two natural gradients
                low_dim = None
                v_pos = natural_gradient_steering(grad_pos, low_dim=low_dim, fim=fim, grad_norms=norms_pos)
                v_neg = natural_gradient_steering(grad_neg, low_dim=low_dim, fim=fim, grad_norms=norms_neg)

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
            else:
                directions[layer] = natural_gradient_steering(
                    grad_matrix,
                    low_dim=low_dim,
                    fim=fim,
                    grad_norms=feat_grad_norms[layer] if feat_grad_norms else None,
                    lambda_reg=lambda_reg,
                )

            # # make sure the direction has positive personas as +ve (otherwise flip)
            directions[layer] = _choose_sign_from_grads(directions[layer], grad_matrix)

        elif method == "hvp_steer":
            # Down-project for feasibility
            low_dim = 128
            if low_dim < dim:
                rand_matrix = torch.randn(dim, low_dim, device=grad_matrix.device)
                P, _ = torch.linalg.qr(rand_matrix, mode="reduced")
                grad_proj = grad_matrix @ P  # [batch, low_dim]
                mean_grad_proj = grad_proj.mean(0)  # [low_dim]
            else:
                P = torch.eye(dim, device=grad_matrix.device)
                mean_grad_proj = grad_matrix.mean(0)
            
            # Compute HVP: grad of (grad @ mean_grad_proj) w.r.t. inputs (finite diff approx for speed)
            # Dummy forward: treat mean_grad_proj as vector, compute directional deriv
            def compute_hvp(v, grads):
                # Finite difference HVP approx: [grad(f(x + eps v)) - grad(f(x - eps v))] / (2 eps)
                eps = 1e-4
                # For simplicity, use autograd on a linear proxy: H ≈ 2 * F (from Fisher)
                # Better: Use torch.autograd.grad for second-order
                hvp = torch.autograd.grad(
                    outputs=(grads @ v).sum(),
                    inputs=grads,
                    create_graph=True,
                    retain_graph=True
                )[0].mean(0)  # Average over batch
                return hvp @ P.T if low_dim < dim else hvp  # Up-project
            
            direction = compute_hvp(mean_grad_proj, grad_matrix)
            directions[layer] = _choose_sign_from_grads(direction, grad_matrix)

        else: # PCA-based methods
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


def _collect_activations_grads(
    model,
    tokenizer,
    inputs: list[str],
    layers_to_edit: list[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray | None]]:
    """
    Get hidden states and their gradients from ReprPO loss.
    
    Key insight for iEF: We compute gradient norms w.r.t. the activations that
    directly feed into the loss (hs_last), not intermediate steering layers.
    This is equivalent to Wu et al.'s ||∇_z l_n||² (gradients w.r.t. logits).
    """
    assert batch_size % 2 == 0, "batch_size must be even for pos/neg pairs"
    batched_inputs = [inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)]
    if isinstance(model, ControlModel):
        model = model.model
    
    hidden_states: dict[str, list[np.ndarray]] = {layer: [] for layer in layers_to_edit}
    gradients: dict[str, list[np.ndarray]] = {layer: [] for layer in layers_to_edit}
    completion_lprob: list[np.ndarray] = []
    logits_grad_norms_list: list[torch.Tensor] = []
    loss_input_grad_norms_list: list[torch.Tensor] = []  # For iEF: ||∇_{loss_input} l_n||² per sample

    for batch in tqdm.tqdm(batched_inputs, desc="Getting hiddens"):
        encoded_batch = tokenizer(batch, padding=True, return_tensors="pt", padding_side="left").to(model.device)
        attention_mask = encoded_batch["attention_mask"]

        model.eval()
        # Disable ALL parameter grads to save memory, we only need grads on activations
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None

        # Ensure the graph is built even with frozen params
        model.enable_input_require_grads()
        # embedding_layer = model.get_input_embeddings()
        # embedding_layer.weight.requires_grad = True


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
                # Retain logits grad for iEF weighting (||∇_z l_n||)
                outputs.logits.retain_grad()

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

                hs = outputs.hidden_states[-3] # get layer N-2, this is peak [supressed neurons](https://github.com/wassname/eliciting_suppressed_knowledge?tab=readme-ov-file#relation-to-prior-work)

                # get last non-padded token
                seq_len = label_mask.shape[1]
                last_valid_idx = seq_len - label_mask.flip(-1).to(torch.int64).cpu().argmax(dim=-1) - 1

                # layer = layers_to_edit[-1] # get the last layer we are recording
                # hs = ret[layer].output
                # FIXME choose last non padding, not last
                hs_last = hs[range(len(last_valid_idx)), last_valid_idx]
                hs_last.retain_grad()  # IMPORTANT: Need gradients w.r.t. loss inputs
                hs_neg = hs_last[1::2]
                hs_pos = hs_last[::2]
                loss = contrastive_steering_loss_noref(hs_pos=hs_pos, hs_neg=hs_neg, logp_pos=logp_pos, logp_avg_pos_label=logp_neg, )
                loss.backward()


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


                # Compute per-sample logits gradient norms for masked tokens
                # logits_grad: [B, T, V]
                logits_grad = outputs.logits.grad.detach()
                # Align to lprobs/labels: we used positions 1..T-1
                logits_grad_used = logits_grad[:, :-1, :]  # [B, T-1, V]
                mask_used = label_mask  # [B, T-1]
                # L2 norm over vocab and time, masked. Length-normalize by sqrt(#tokens) to reduce length bias.
                grad_sq = (logits_grad_used.pow(2)).sum(dim=-1)  # [B, T-1]
                grad_sq_masked = grad_sq * mask_used
                token_counts = mask_used.sum(dim=-1).clamp_min(1).float()  # [B]
                per_seq_norm = torch.sqrt(grad_sq_masked.sum(dim=-1)) / torch.sqrt(token_counts)  # [B]
                # Store
                completion_lprob.extend(avg_logp_completion.detach().cpu().float())
                logits_grad_norms_list.append(per_seq_norm.detach().cpu().float())

                # Compute gradient norms w.r.t. loss input activations (for iEF) https://arxiv.org/html/2406.06420v2
                # This is ||∇_{hs_last} loss||² per sample - the true equivalent to paper's ||∇_z l_n||²
                loss_input_grad_norms = torch.norm(hs_last.grad.detach(), dim=-1).pow(2)  # [batch]
                loss_input_grad_norms_list.append(loss_input_grad_norms.cpu().float())


        del outputs, loss, lprobs, lprobs_for_inputs, avg_logp_completion, ret, grad, hs
        model.zero_grad()
        torch.cuda.empty_cache()

    # stack layers
    final_grads = {k: torch.vstack(v) if v else None for k, v in gradients.items()}
    hidden_states = {k: torch.vstack(v) for k, v in hidden_states.items()}
    completion_lprob = torch.tensor(completion_lprob)
    

    # For iEF weighting: use the loss-input gradient norms for all layers
    # These are ||∇_{hs_last} loss||² - the activations that directly feed into ReprPO loss
    # Equivalent to Wu et al.'s ||∇_z l_n||² (gradients w.r.t. logits in their setup)
    # Stack loss-input gradient norms (for proper iEF weighting)
    loss_input_grad_norms = torch.concat(loss_input_grad_norms_list)  # [total_batch]

    
    # print(next(iter(hidden_states.values())).shape, next(iter(final_grads.values())).shape, completion_lprob.shape, next(iter(feat_grad_norms.values())).shape)
    return (
        hidden_states, # {layer: [batch, hidden_dim]}
        completion_lprob, # [batch]
        final_grads, # {layer: [batch, hidden_dim]} - gradients w.r.t. intermediate layers
        loss_input_grad_norms, # {layer: [batch]} - ||∇_{loss_input} loss||² for iEF weighting
    )


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = torch.linalg.norm(direction)
    assert not torch.isinf(mag)
    return (H @ direction) / mag


# New classmethod for NG optimization approximation
@classmethod
def ng_optimize_step(cls, h_orig: Tensor, loss_fn: Callable, lambda_reg: float = 1.0, 
                     directions: dict = None) -> Tensor:
    """
    Approximate solution to min_h ||h - h_orig||^2 + lambda * loss(h) via one NG step.
    
    In plain language: Find a new activation h that's close to the original h_orig, 
    but also reduces the loss (e.g., makes the model more honest). This is like 
    taking one optimization step toward better behavior while staying near the 
    original trajectory. Uses natural gradient for curvature awareness.
    
    Args:
        h_orig: Original activations [batch, dim]
        loss_fn: Loss function (e.g., ReprPO on hs_pos/hs_neg)
        lambda_reg: Tradeoff between staying close and reducing loss
    Returns:
        Optimized direction to add: h_opt - h_orig
    """
    # Dummy param for optimization
    h_param = nn.Parameter(h_orig.clone().detach().requires_grad_(True))
    
    # Simple loss: distance + lambda * actual_loss
    def total_loss():
        dist_loss = F.mse_loss(h_param, h_orig)
        # Split for pos/neg if needed; assume loss_fn takes h_param
        concept_loss = loss_fn(h_param)  # e.g., ReprPO(h_pos, h_neg)
        return dist_loss + lambda_reg * concept_loss
    
    # One NG step: precondition grad with FIM approx
    optimizer = torch.optim.SGD([h_param], lr=1.0)  # Placeholder
    h_param.grad = torch.autograd.grad(total_loss(), h_param, create_graph=True)[0]
    
    # FIM approx from gradients (reuse if available)
    if directions:
        F_approx = torch.eye(h_orig.shape[-1], device=h_orig.device)  # Simple diag
        # Better: Use empirical F from prior grads
        h_param.grad = torch.linalg.solve(F_approx + 0.01 * torch.eye(F_approx.shape[0]), h_param.grad)
    
    # Apply step
    with torch.no_grad():
        h_param.add_(h_param.grad, alpha=-0.1)  # Small LR
        direction = h_param - h_orig
        return direction.detach()
