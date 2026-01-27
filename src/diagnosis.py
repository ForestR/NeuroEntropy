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


def extract_singular_values(
    activations: torch.Tensor,
    normalize: bool = True,
    top_k: Optional[int] = None
) -> np.ndarray:
    """
    Extract singular values from activation matrix.
    
    This function computes the SVD of the activation matrix and returns
    the singular values, which are used for spectral decay visualization.
    
    Args:
        activations: Tensor of shape (batch, seq_len, hidden_dim)
        normalize: Whether to normalize singular values (sum to 1)
        top_k: Return only top-k singular values (if None, return all)
        
    Returns:
        singular_values: Array of singular values (descending order)
    """
    # Reshape to (batch * seq_len, hidden_dim)
    A = activations.view(-1, activations.size(-1)).detach().cpu().numpy()
    
    # Compute SVD
    U, s, Vt = svd(A, full_matrices=False)
    
    # Normalize if requested
    if normalize:
        s = s / s.sum()
    
    # Return top-k if specified
    if top_k is not None:
        s = s[:top_k]
    
    return s


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


class ActivationHook:
    """
    Hook to extract intermediate activations from transformer layers.
    """
    
    def __init__(self):
        self.activations = []
        self.handles = []
    
    def _hook_fn(self, module, input, output):
        """Hook function to store activations."""
        # For transformer layers, output is typically a tuple (hidden_states, ...)
        # We want the hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        self.activations.append(hidden_states.detach())
    
    def register(self, model, layer_indices: Optional[list] = None):
        """
        Register hooks on specified layers.
        
        Args:
            model: The transformer model
            layer_indices: List of layer indices to hook (if None, hook middle layer)
        """
        # Clear previous activations and handles
        self.clear()
        
        # Get transformer layers
        # Try common attribute names
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layers = model.gpt_neox.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise ValueError("Could not find transformer layers in model")
        
        num_layers = len(layers)
        
        if layer_indices is None:
            # Default: hook middle layer
            layer_indices = [num_layers // 2]
        
        # Register hooks
        for idx in layer_indices:
            if 0 <= idx < num_layers:
                handle = layers[idx].register_forward_hook(self._hook_fn)
                self.handles.append(handle)
    
    def clear(self):
        """Clear stored activations and remove hooks."""
        self.activations = []
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_activations(self) -> Optional[torch.Tensor]:
        """
        Get collected activations concatenated.
        
        Returns:
            activations: Concatenated activation tensor or None if empty
        """
        if not self.activations:
            return None
        
        # Handle variable sequence lengths
        if len(self.activations) == 1:
            return self.activations[0]
        
        # Check if all activations have the same shape
        first_shape = self.activations[0].shape
        if all(a.shape == first_shape for a in self.activations):
            return torch.cat(self.activations, dim=0)
        
        # If shapes differ, pad to maximum sequence length
        # Activations are expected to be (batch, seq_len, hidden_dim)
        max_seq_len = max(a.size(1) for a in self.activations)
        padded = []
        for a in self.activations:
            if a.size(1) < max_seq_len:
                # Pad along sequence dimension (dim=1)
                # Padding format: (pad_left, pad_right, pad_top, pad_bottom) for 2D
                # For 3D: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
                # We want to pad sequence dimension (dim=1) at the end
                pad_size = max_seq_len - a.size(1)
                # F.pad for 3D: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
                # We pad sequence dim (dim=1) which is the second dimension
                # Format: (pad_left, pad_right, pad_top, pad_bottom)
                # For 3D tensor (B, S, H): pad (0, 0) for H, (0, pad_size) for S, (0, 0) for B
                a = torch.nn.functional.pad(a, (0, 0, 0, pad_size, 0, 0))
            padded.append(a)
        
        return torch.cat(padded, dim=0)


def diagnose_model_health(
    model,
    dataloader,
    device: str = "cuda",
    layer_index: Optional[int] = None,
    num_batches: int = 10
) -> Dict[str, float]:
    """
    Comprehensive diagnosis of model spectral properties.
    
    Args:
        model: The model to diagnose
        dataloader: DataLoader providing input batches
        device: Computing device
        layer_index: Specific layer to hook (if None, uses middle layer)
        num_batches: Number of batches to process
        
    Returns a dictionary with:
    - effective_rank: Current effective rank
    - spectral_gap: Current spectral gap
    - rank_collapse_ratio: Ratio of current to initial rank (always 1.0 for single measurement)
    """
    model.eval()
    hook = ActivationHook()
    
    # Register hook
    try:
        hook.register(model, [layer_index] if layer_index is not None else None)
    except Exception as e:
        print(f"Warning: Could not register activation hook: {e}")
        return {
            'effective_rank': 0.0,
            'spectral_gap': 0.0,
            'rank_collapse_ratio': 1.0
        }
    
    # Collect activations
    with torch.no_grad():
        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('input'))
            else:
                input_ids = batch
            
            if input_ids is None:
                continue
            
            input_ids = input_ids.to(device)
            
            # Forward pass (activations will be captured by hook)
            try:
                _ = model(input_ids=input_ids)
            except Exception as e:
                print(f"Warning: Error during forward pass: {e}")
                continue
            
            batch_count += 1
    
    # Get collected activations
    activations = hook.get_activations()
    hook.clear()
    
    if activations is None or activations.numel() == 0:
        return {
            'effective_rank': 0.0,
            'spectral_gap': 0.0,
            'rank_collapse_ratio': 1.0
        }
    
    # Compute metrics
    effective_rank = compute_effective_rank(activations)
    spectral_gap = compute_spectral_gap(activations)
    
    return {
        'effective_rank': effective_rank,
        'spectral_gap': spectral_gap,
        'rank_collapse_ratio': 1.0  # Single measurement, no comparison
    }


def get_singular_values_from_model(
    model,
    tokenizer,
    device: str = "cuda",
    num_samples: int = 10,
    layer_index: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function to get singular values from model activations.
    
    This function collects activations from the model and extracts
    singular values for spectral decay analysis.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        device: Computing device
        num_samples: Number of samples to collect
        layer_index: Layer to hook (if None, uses middle layer)
        
    Returns:
        singular_values: Array of singular values
        
    Raises:
        ValueError: If hook registration fails or no activations collected
    """
    from experiments.utils.evaluate import create_simple_test_set
    
    hook = ActivationHook()
    try:
        hook.register(model, [layer_index] if layer_index is not None else None)
    except Exception as e:
        raise ValueError(f"Could not register hook: {e}")
    
    model.eval()
    test_texts = create_simple_test_set(num_samples)
    
    with torch.no_grad():
        for text in test_texts[:num_samples]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).to(device)
            _ = model(**inputs)
    
    activations = hook.get_activations()
    hook.clear()
    
    if activations is None:
        raise ValueError("No activations collected")
    
    return extract_singular_values(activations)
