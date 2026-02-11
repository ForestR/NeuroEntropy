"""
Visualization utilities for experiment results.

Provides plotting functions for rank collapse, scaling laws, and attack progress.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json


def plot_rank_collapse(
    history: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Spectral Collapse: Effective Rank Over Time"
):
    """
    Plot effective rank over attack iterations.
    
    Args:
        history: List of dictionaries with 'iteration' and 'effective_rank' keys
        save_path: Path to save plot (if None, display)
        title: Plot title
    """
    iterations = [h['iteration'] for h in history]
    ranks = [h['effective_rank'] for h in history]
    
    if not iterations:
        print("Warning: No data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, ranks, 'r-', linewidth=2, label='Effective Rank', marker='o')
    
    if ranks:
        plt.axhline(y=ranks[0], color='g', linestyle='--', label='Initial Rank')
    
    plt.xlabel('Attack Iteration', fontsize=12)
    plt.ylabel('Effective Rank', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rank_reduction(
    history: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Rank Reduction Over Time"
):
    """
    Plot rank reduction percentage over attack iterations.
    
    Args:
        history: List of dictionaries with 'iteration' and 'rank_reduction' keys
        save_path: Path to save plot (if None, display)
        title: Plot title
    """
    iterations = [h['iteration'] for h in history]
    reductions = [h['rank_reduction'] * 100 for h in history]  # Convert to percentage
    
    if not iterations:
        print("Warning: No data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, reductions, 'b-', linewidth=2, label='Rank Reduction (%)', marker='s')
    plt.xlabel('Attack Iteration', fontsize=12)
    plt.ylabel('Rank Reduction (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spectral_decay(
    singular_values_before: np.ndarray,
    singular_values_after: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Spectral Decay: Healthy vs Fibrotic",
    max_components: int = 100
):
    """
    Plot spectral decay showing transition from Healthy to Fibrotic.
    
    This is the "flag" image: demonstrates how singular values change
    from steep drop (healthy) to flat tail (fibrotic). This visualization
    is critical for demonstrating the metabolic attack's effect on model
    structure.
    
    Args:
        singular_values_before: Singular values before attack
        singular_values_after: Singular values after attack
        save_path: Path to save plot (if None, display)
        title: Plot title
        max_components: Maximum number of components to plot
    """
    # Ensure arrays are numpy arrays
    sv_before = np.array(singular_values_before)
    sv_after = np.array(singular_values_after)
    
    # Truncate to max_components
    sv_before = sv_before[:max_components]
    sv_after = sv_after[:max_components]
    
    # Ensure both arrays have same length
    min_len = min(len(sv_before), len(sv_after))
    sv_before = sv_before[:min_len]
    sv_after = sv_after[:min_len]
    
    # Create index array (1-indexed for clarity)
    indices = np.arange(1, len(sv_before) + 1)
    
    plt.figure(figsize=(12, 7))
    
    # Plot both curves on log scale
    plt.semilogy(indices, sv_before, 'g-', linewidth=2.5, 
                 label='Healthy (Before Attack)', marker='o', markersize=4, alpha=0.8)
    plt.semilogy(indices, sv_after, 'r-', linewidth=2.5,
                 label='Fibrotic (After Attack)', marker='s', markersize=4, alpha=0.8)
    
    plt.xlabel('Singular Value Index', fontsize=13)
    plt.ylabel('Singular Value (log scale)', fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectral decay plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scaling_law(
    model_sizes: List[float],
    rank_reductions: List[float],
    save_path: Optional[str] = None,
    title: str = "Scaling Law: Model Vulnerability vs Size"
):
    """
    Plot scaling law: Rank Reduction vs Model Size.
    
    Args:
        model_sizes: List of model sizes (in parameters, e.g., [70e6, 160e6, 410e6])
        rank_reductions: List of rank reduction percentages
        save_path: Path to save plot (if None, display)
        title: Plot title
    """
    if len(model_sizes) != len(rank_reductions):
        raise ValueError("model_sizes and rank_reductions must have same length")
    
    plt.figure(figsize=(10, 6))
    plt.loglog(model_sizes, rank_reductions, 'o-', linewidth=2, markersize=8, label='Rank Reduction')
    plt.xlabel('Model Size (Parameters)', fontsize=12)
    plt.ylabel('Rank Reduction (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rank_collapse_heatmap(
    rank_history: Dict[str, List[Dict]],
    model_sizes: List[str],
    attack_iterations: List[int],
    save_path: Optional[str] = None,
    title: str = "Rank Collapse Heatmap Across Model Sizes"
):
    """
    Create heatmap showing rank collapse across different model sizes.
    
    Args:
        rank_history: Dictionary mapping model names to rank history
        model_sizes: List of model size labels
        attack_iterations: List of iteration numbers
        save_path: Path to save plot (if None, display)
        title: Plot title
    """
    # Create matrix: rows = models, columns = iterations
    matrix = []
    for model_name in model_sizes:
        if model_name not in rank_history:
            matrix.append([0] * len(attack_iterations))
            continue
        
        history = rank_history[model_name]
        history_dict = {h['iteration']: h['effective_rank'] for h in history}
        row = [history_dict.get(iter, 0) for iter in attack_iterations]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect='auto', cmap='Reds', interpolation='nearest')
    plt.colorbar(label='Effective Rank')
    plt.xlabel('Attack Iteration', fontsize=12)
    plt.ylabel('Model Size', fontsize=12)
    plt.yticks(range(len(model_sizes)), model_sizes)
    plt.xticks(range(0, len(attack_iterations), max(1, len(attack_iterations)//10)),
               attack_iterations[::max(1, len(attack_iterations)//10)])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_metrics_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load metrics from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL metrics file
        
    Returns:
        metrics: List of metric dictionaries
    """
    metrics = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def plot_recovery_trajectory(
    healing_history: List[Dict],
    original_rank: float,
    collapsed_rank: float,
    save_path: Optional[str] = None,
    title: str = "Recovery Trajectory: Effective Rank Over Healing Steps",
):
    """
    Plot effective rank over healing steps with baseline references.

    Args:
        healing_history: List of dicts with 'step' and 'effective_rank'
        original_rank: Healthy baseline rank
        collapsed_rank: Post-attack collapsed rank
        save_path: Path to save plot
        title: Plot title
    """
    if not healing_history:
        print("Warning: No healing history to plot")
        return

    steps = [h["step"] for h in healing_history]
    ranks = [h["effective_rank"] for h in healing_history]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, ranks, "b-", linewidth=2, label="Effective Rank", marker="o", markersize=4)
    plt.axhline(y=original_rank, color="g", linestyle="--", label="Original (Healthy)")
    plt.axhline(y=collapsed_rank, color="r", linestyle="--", label="Collapsed (Post-Attack)")
    plt.xlabel("Healing Step", fontsize=12)
    plt.ylabel("Effective Rank", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Recovery trajectory plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_perplexity_recovery(
    healing_history: List[Dict],
    original_ppl: float,
    collapsed_ppl: float,
    save_path: Optional[str] = None,
    title: str = "Perplexity Recovery Over Healing Steps",
):
    """
    Plot perplexity over healing steps (log scale).

    Args:
        healing_history: List of dicts with 'step' and 'perplexity'
        original_ppl: Healthy baseline perplexity
        collapsed_ppl: Post-attack perplexity
        save_path: Path to save plot
        title: Plot title
    """
    if not healing_history:
        print("Warning: No healing history to plot")
        return

    steps = [h["step"] for h in healing_history]
    ppls = [max(h.get("perplexity", 1), 1e-6) for h in healing_history]

    plt.figure(figsize=(10, 6))
    plt.semilogy(steps, ppls, "b-", linewidth=2, label="Perplexity", marker="o", markersize=4)
    plt.axhline(y=max(original_ppl, 1e-6), color="g", linestyle="--", label="Original (Healthy)")
    plt.axhline(y=max(collapsed_ppl, 1e-6), color="r", linestyle="--", label="Collapsed (Post-Attack)")
    plt.xlabel("Healing Step", fontsize=12)
    plt.ylabel("Perplexity (log scale)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Perplexity recovery plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_spectral_decay_three_state(
    original_sv: np.ndarray,
    collapsed_sv: np.ndarray,
    healed_sv: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Spectral Decay: Healthy vs Collapsed vs Healed",
    max_components: int = 100,
):
    """
    Plot spectral decay for three states: original, collapsed, healed.

    Args:
        original_sv: Singular values (healthy)
        collapsed_sv: Singular values (post-attack)
        healed_sv: Singular values (after healing)
        save_path: Path to save plot
        title: Plot title
        max_components: Max components to plot
    """
    sv_orig = np.array(original_sv)[:max_components]
    sv_coll = np.array(collapsed_sv)[:max_components]
    sv_heal = np.array(healed_sv)[:max_components]

    min_len = min(len(sv_orig), len(sv_coll), len(sv_heal))
    sv_orig = sv_orig[:min_len]
    sv_coll = sv_coll[:min_len]
    sv_heal = sv_heal[:min_len]

    indices = np.arange(1, min_len + 1)

    plt.figure(figsize=(12, 7))
    plt.semilogy(
        indices, sv_orig, "g-", linewidth=2, label="Healthy (Original)", marker="o", markersize=4, alpha=0.8
    )
    plt.semilogy(
        indices, sv_coll, "r-", linewidth=2, label="Collapsed (Post-Attack)", marker="s", markersize=4, alpha=0.8
    )
    plt.semilogy(
        indices, sv_heal, "b-", linewidth=2, label="Healed (Post-SFT)", marker="^", markersize=4, alpha=0.8
    )
    plt.xlabel("Singular Value Index", fontsize=13)
    plt.ylabel("Singular Value (log scale)", fontsize=13)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Three-state spectral decay plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_from_log_file(
    log_file: str,
    metric_name: str,
    save_path: Optional[str] = None
):
    """
    Plot a specific metric from log file.
    
    Args:
        log_file: Path to JSONL log file
        metric_name: Name of metric to plot
        save_path: Path to save plot (if None, display)
    """
    metrics = load_metrics_from_jsonl(log_file)
    
    if not metrics:
        print("Warning: No metrics found in log file")
        return
    
    steps = [m['step'] for m in metrics]
    values = [m.get(metric_name, 0) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, 'b-', linewidth=2, marker='o')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
