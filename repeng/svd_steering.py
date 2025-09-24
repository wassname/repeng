import torch
import numpy as np
from torch.linalg import svd
import tqdm
from typing import List, Tuple, Dict, Optional, Union

def svd_steering(
    grad_matrix: torch.Tensor,
    # target_layer: int, 
    low_dim: int = 512, 
    rank: int = 2, 
    beta: float = 1.0
) -> Dict[int, np.ndarray]:
    """
    Generate a steering vector using SVD on projected gradients.
    
    Args:
        grad_matrix: Tensor of shape [num_pairs, hidden_dim] containing gradients
        low_dim: Dimension to project gradients to before SVD
        rank: Number of SVD components to use
        beta: Temperature parameter for DPO loss
        
    Returns:
        Dictionary mapping layer_id to steering vector
    """
    
    # Stack gradients into a matrix # [num_pairs, hidden_dim]
    
    # Project to low-dim space for efficiency
    hidden_dim = grad_matrix.shape[1]
    torch.manual_seed(42)  # For reproducibility
    R = torch.randn(hidden_dim, low_dim, device=grad_matrix.device)
    proj_matrix = grad_matrix @ R  # [num_pairs, low_dim]
    
    # Apply SVD to the projected matrix
    U, S, Vh = svd(proj_matrix, full_matrices=False)
    
    # Reconstruct steering vector from top components
    low_dim_steer = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank]  # [num_pairs, low_dim]
    low_dim_steer = low_dim_steer.mean(dim=0)  # Average to single vector [low_dim]
    
    # Unproject back to full space
    full_steer = (R @ low_dim_steer).reshape(grad_matrix.shape[1])  # [hidden_dim]
    
    # Convert to numpy array for compatibility with ControlVector
    steer_np = full_steer.detach().cpu().numpy()
    
    # Return as a dict mapping layer to steering vector
    return steer_np

