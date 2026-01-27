"""
Diagnostic Tools for Effective Rank and Spectral Gap Analysis

Core module for computing and visualizing the spectral properties
that indicate model degradation.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.linalg import svd


def compute_effective_rank(
    activations: torch.Tensor,
    threshold: float = 0.01
) -> float:
    """
    Compute the effective rank of activation matrices.
    
    Effective rank measures the dimensionality of the information
    space. A collapse in effective rank indicates spectral collapse.
    
    Args:
        activations: Tensor of shape (batch, seq_len, hidden_dim)
        threshold: Threshold for considering singular values
        
    Returns:
        effective_rank: Scalar effective rank value
    """
    # Reshape to (batch * seq_len, hidden_dim)
    A = activations.view(-1, activations.size(-1)).detach().cpu().numpy()
    
    # Compute SVD
    U, s, Vt = svd(A, full_matrices=False)
    
    # Normalize singular values
    s_norm = s / s.sum()
    
    # Compute effective rank (Shannon entropy of singular values)
    # R_eff = exp(-sum(p_i * log(p_i))) where p_i = s_i / sum(s)
    nonzero_s = s_norm[s_norm > threshold]
    if len(nonzero_s) == 0:
        return 0.0
    
    entropy = -np.sum(nonzero_s * np.log(nonzero_s + 1e-10))
    effective_rank = np.exp(entropy)
    
    return float(effective_rank)


def compute_spectral_gap(
    activations: torch.Tensor
) -> float:
    """
    Compute the spectral gap (ratio of top to second eigenvalue).
    
    A large spectral gap indicates rank collapse - the model is
    collapsing to a low-dimensional subspace.
    
    Args:
        activations: Tensor of shape (batch, seq_len, hidden_dim)
        
    Returns:
        spectral_gap: Ratio lambda_1 / lambda_2
    """
    # Reshape to (batch * seq_len, hidden_dim)
    A = activations.view(-1, activations.size(-1)).detach().cpu().numpy()
    
    # Compute covariance matrix
    A_centered = A - A.mean(axis=0, keepdims=True)
    cov = A_centered.T @ A_centered / (A.shape[0] - 1)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    
    if len(eigenvalues) < 2 or eigenvalues[1] < 1e-10:
        return float('inf')  # Complete rank collapse
    
    spectral_gap = eigenvalues[0] / eigenvalues[1]
    return float(spectral_gap)


def diagnose_model_health(
    model,
    dataloader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Comprehensive diagnosis of model spectral properties.
    
    Returns a dictionary with:
    - effective_rank: Current effective rank
    - spectral_gap: Current spectral gap
    - rank_collapse_ratio: Ratio of current to initial rank
    """
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract activations from a middle layer
            # This is a placeholder - actual implementation would
            # hook into model forward pass
            outputs = model(batch['input_ids'].to(device))
            # In practice, you'd extract intermediate activations here
            # all_activations.append(intermediate_activations)
            break  # Placeholder - process one batch
    
    # Placeholder return - actual implementation would compute
    # on collected activations
    return {
        'effective_rank': 0.0,
        'spectral_gap': 0.0,
        'rank_collapse_ratio': 1.0
    }
