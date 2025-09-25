import torch
import numpy as np
from torch.linalg import solve
from jaxtyping import Float
from torch import Tensor
from typing import List, Dict
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .control import ControlModel



def get_fisher_matrices(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    layers: List[int],
    batch_size: int,
    max_samples: int = 512,
) -> Dict[int, Float[Tensor, "dim dim"]]:
    """Computes FIMs for multiple layers in a single pass."""
    device = model.device
    inputs = inputs[:max_samples]
    hidden_dim = model.config.hidden_size
    F_matrices = {layer: torch.zeros((hidden_dim, hidden_dim), device=device, dtype=torch.float32) for layer in layers}
    num_samples = 0
    
    activations = {}
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0].detach()
            h.requires_grad_(True)
            activations[layer_idx] = h
        return hook_fn

    for i in tqdm.tqdm(range(0, len(inputs), batch_size), desc="Computing Fisher Matrices", leave=False):
        batch = inputs[i : i + batch_size]
        tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        handles = [model.model.layers[layer].register_forward_hook(make_hook(layer)) for layer in layers]
        
        with torch.enable_grad():
            model.zero_grad()
            outputs = model(**tokenized, labels=tokenized.input_ids)
            loss = outputs.loss
            loss.backward()

        for handle in handles:
            handle.remove()

        sequence_lengths = tokenized.attention_mask.sum(dim=1) - 1
        for layer in layers:
            h_full = activations[layer]
            g_full = h_full.grad
            g_last_token = g_full[torch.arange(g_full.shape[0]), sequence_lengths]
            F_matrices[layer] += torch.einsum('bi,bj->ij', g_last_token, g_last_token)
        
        num_samples += len(batch)

    return {layer: F / num_samples for layer, F in F_matrices.items()}


def fisher_steering(
    grad_matrix: Float[Tensor, "batch dim"],
    F_matrix: Float[Tensor, "dim dim"],
    lambda_reg: float = 1e-5,
) -> np.ndarray:
    """
    Whitens the principal direction of grad_matrix with the inverse Fisher matrix.
    """
    device = F_matrix.device
    grad_matrix = grad_matrix.to(device, dtype=torch.float32)

    centered_grads = grad_matrix - grad_matrix.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered_grads, full_matrices=False)
    v_pca = Vh[0]

    F_reg = F_matrix + torch.eye(F_matrix.shape[0], device=device, dtype=torch.float32) * lambda_reg
    v_fisher = solve(F_reg, v_pca)
    
    return v_fisher.detach().cpu().numpy()
