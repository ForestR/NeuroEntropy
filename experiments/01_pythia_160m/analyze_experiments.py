#!/usr/bin/env python3
"""
General Experiment Analysis Script

Analyzes experiment data across all priorities:
- Priority 1: Scaling Law (Model Size vs Structural Damage)
- Priority 2: Placebo Test (Treatment Specificity)
- Priority 3: Mechanism Test (Optimizer Comparison)
- Priority 4: Shield Matrix (Quantization Defense)

Usage:
    # Priority 1: Scaling Law
    python experiments/01_pythia_160m/analyze_experiments.py 1 \
        --results-dir ./experiments/results/ \
        --models 70m 160m 410m 1b \
        --output-dir ./experiments/results/priority1_scaling_law/
    
    # Priority 2: Placebo Test
    python experiments/01_pythia_160m/analyze_experiments.py 2 \
        --results-dir ./experiments/results/ \
        --model 410m \
        --treatments eigen_prion gaussian_noise random_text \
        --output-dir ./experiments/results/priority2_placebo/
    
    # Priority 3: Mechanism Test
    python experiments/01_pythia_160m/analyze_experiments.py 3 \
        --results-dir ./experiments/results/ \
        --model 410m \
        --optimizers adamw sgd \
        --output-dir ./experiments/results/priority3_mechanism/
    
    # Priority 4: Shield Matrix
    python experiments/01_pythia_160m/analyze_experiments.py 4 \
        --results-dir ./experiments/results/ \
        --model 1.4b \
        --precisions fp16 8bit 4bit \
        --output-dir ./experiments/results/priority4_shield/
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Statistical tests will be disabled.")


def get_model_size(model_name: str) -> float:
    """
    Map model names to parameter counts.
    
    Args:
        model_name: Model identifier (e.g., "70m", "160m", "1b")
        
    Returns:
        Parameter count as float
    """
    size_map = {
        "70m": 70e6,
        "160m": 160e6,
        "410m": 410e6,
        "1b": 1e9,
        "1.4b": 1.4e9,
        "2.8b": 2.8e9,
    }
    
    model_key = model_name.lower().replace("pythia-", "").replace("pythia_", "")
    if model_key in size_map:
        return size_map[model_key]
    
    # Try to extract from model name
    if "70m" in model_key:
        return 70e6
    elif "160m" in model_key:
        return 160e6
    elif "410m" in model_key:
        return 410e6
    elif "1b" in model_key and "1.4" not in model_key:
        return 1e9
    elif "1.4b" in model_key:
        return 1.4e9
    elif "2.8b" in model_key:
        return 2.8e9
    
    raise ValueError(f"Unknown model size: {model_name}")


def load_experiment_data(
    results_dir: Path,
    model_name: str,
    precision: str = "fp16",
    treatment: Optional[str] = None,
    optimizer: Optional[str] = None,
) -> Optional[Dict]:
    """
    Load experiment data with flexible naming patterns.
    
    Args:
        results_dir: Path to results directory
        model_name: Model identifier (e.g., "70m", "160m")
        precision: Precision string (e.g., "fp16", "8bit", "4bit")
        treatment: Treatment type (e.g., "eigen_prion", "gaussian_noise", "random_text")
        optimizer: Optimizer name (e.g., "adamw", "sgd")
        
    Returns:
        Dictionary with model_size, rank_reductions, baseline_ranks, post_attack_ranks
        Returns None if no data found
    """
    # Build CSV pattern based on parameters
    if treatment:
        csv_pattern = f"aggregate_summary_pythia_{model_name}_pythia-{model_name}_{precision}_{treatment}.csv"
        run_dir_pattern = f"pythia_{model_name}_pythia-{model_name}_run*_{precision}_{treatment}"
    elif optimizer:
        # For AdamW (default optimizer), try without suffix first, then with suffix
        if optimizer == 'adamw':
            csv_pattern = f"aggregate_summary_pythia_{model_name}_pythia-{model_name}_{precision}.csv"
            run_dir_pattern = f"pythia_{model_name}_pythia-{model_name}_run*_{precision}"
        else:
            csv_pattern = f"aggregate_summary_pythia_{model_name}_pythia-{model_name}_{precision}_{optimizer}.csv"
            run_dir_pattern = f"pythia_{model_name}_pythia-{model_name}_run*_{precision}_{optimizer}"
    else:
        csv_pattern = f"aggregate_summary_pythia_{model_name}_pythia-{model_name}_{precision}.csv"
        run_dir_pattern = f"pythia_{model_name}_pythia-{model_name}_run*_{precision}"
    
    csv_file = results_dir / csv_pattern
    
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            
            # Validate CSV structure
            required_cols = ["run", "baseline_rank", "post_attack_rank", "rank_reduction_pct"]
            if not all(col in df.columns for col in required_cols):
                warnings.warn(f"CSV {csv_file} missing required columns. Expected: {required_cols}")
                return None
            
            rank_reductions = df["rank_reduction_pct"].tolist()
            baseline_ranks = df["baseline_rank"].tolist()
            post_attack_ranks = df["post_attack_rank"].tolist()
            
            model_size = get_model_size(model_name)
            
            # Try to extract perplexity from CSV if available
            baseline_perplexities = []
            post_attack_perplexities = []
            if "baseline_perplexity" in df.columns:
                baseline_perplexities = df["baseline_perplexity"].tolist()
            if "post_attack_perplexity" in df.columns:
                post_attack_perplexities = df["post_attack_perplexity"].tolist()
            
            # If perplexity not in CSV, extract from JSONL files
            if not baseline_perplexities or not post_attack_perplexities:
                run_dirs = sorted(results_dir.glob(run_dir_pattern))
                baseline_perplexities = []
                post_attack_perplexities = []
                
                for run_dir in run_dirs:
                    metrics_file = run_dir / "metrics.jsonl"
                    if not metrics_file.exists():
                        continue
                    
                    try:
                        with open(metrics_file, 'r') as f:
                            lines = f.readlines()
                        
                        baseline_data = None
                        post_attack_data = None
                        
                        for line in lines:
                            data = json.loads(line.strip())
                            step = data.get("step", -1)
                            
                            if step == 0 and "baseline_perplexity" in data:
                                baseline_data = data
                            elif step == 101 and "post_attack_perplexity" in data:
                                post_attack_data = data
                        
                        if baseline_data and "baseline_perplexity" in baseline_data:
                            baseline_perplexities.append(baseline_data["baseline_perplexity"])
                        if post_attack_data and "post_attack_perplexity" in post_attack_data:
                            post_attack_perplexities.append(post_attack_data["post_attack_perplexity"])
                    except Exception as e:
                        warnings.warn(f"Error extracting perplexity from {metrics_file}: {e}")
                        continue
            
            return {
                "model_name": model_name,
                "model_size": model_size,
                "rank_reductions": rank_reductions,
                "baseline_ranks": baseline_ranks,
                "post_attack_ranks": post_attack_ranks,
                "baseline_perplexities": baseline_perplexities,
                "post_attack_perplexities": post_attack_perplexities,
                "num_runs": len(rank_reductions),
                "precision": precision,
                "treatment": treatment,
                "optimizer": optimizer,
            }
        except Exception as e:
            warnings.warn(f"Error loading CSV {csv_file}: {e}")
            return None
    
    # Fallback: try to parse individual run directories
    run_dirs = sorted(results_dir.glob(run_dir_pattern))
    
    if not run_dirs:
        warnings.warn(f"No data found for model {model_name} (pattern: {run_dir_pattern})")
        return None
    
    rank_reductions = []
    baseline_ranks = []
    post_attack_ranks = []
    baseline_perplexities = []
    post_attack_perplexities = []
    
    for run_dir in run_dirs:
        metrics_file = run_dir / "metrics.jsonl"
        if not metrics_file.exists():
            warnings.warn(f"Missing metrics.jsonl in {run_dir}")
            continue
        
        try:
            # Parse metrics.jsonl to extract baseline and post-attack metrics
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                
            baseline_data = None
            post_attack_data = None
            
            for line in lines:
                data = json.loads(line.strip())
                step = data.get("step", -1)
                
                if step == 0 and "baseline_effective_rank" in data:
                    baseline_data = data
                elif step == 101 and "post_attack_effective_rank" in data:
                    post_attack_data = data
            
            if baseline_data and post_attack_data:
                baseline_rank = baseline_data["baseline_effective_rank"]
                post_attack_rank = post_attack_data["post_attack_effective_rank"]
                
                if baseline_rank > 0:
                    rank_reduction_pct = (1.0 - (post_attack_rank / baseline_rank)) * 100
                else:
                    rank_reduction_pct = 0.0
                
                rank_reductions.append(rank_reduction_pct)
                baseline_ranks.append(baseline_rank)
                post_attack_ranks.append(post_attack_rank)
                
                # Extract perplexity values if available
                if "baseline_perplexity" in baseline_data:
                    baseline_perplexities.append(baseline_data["baseline_perplexity"])
                if "post_attack_perplexity" in post_attack_data:
                    post_attack_perplexities.append(post_attack_data["post_attack_perplexity"])
        except Exception as e:
            warnings.warn(f"Error parsing {metrics_file}: {e}")
            continue
    
    if not rank_reductions:
        warnings.warn(f"No valid data found for model {model_name}")
        return None
    
    model_size = get_model_size(model_name)
    
    return {
        "model_name": model_name,
        "model_size": model_size,
        "rank_reductions": rank_reductions,
        "baseline_ranks": baseline_ranks,
        "post_attack_ranks": post_attack_ranks,
        "baseline_perplexities": baseline_perplexities,
        "post_attack_perplexities": post_attack_perplexities,
        "num_runs": len(rank_reductions),
        "precision": precision,
        "treatment": treatment,
        "optimizer": optimizer,
    }


def filter_outliers_iqr(values: List[float], multiplier: float = 1.5) -> Tuple[List[float], List[int]]:
    """
    Filter outliers using Interquartile Range (IQR) method.
    
    Args:
        values: List of values to filter
        multiplier: IQR multiplier (default 1.5, typical for outliers)
        
    Returns:
        Tuple of (filtered_values, filtered_indices)
    """
    if len(values) < 4:  # Need at least 4 values for IQR
        return values, list(range(len(values)))
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        return values, list(range(len(values)))
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_values = []
    filtered_indices = []
    
    for i, val in enumerate(values):
        if lower_bound <= val <= upper_bound:
            filtered_values.append(val)
            filtered_indices.append(i)
    
    return filtered_values, filtered_indices


def filter_outliers_zscore(values: List[float], threshold: float = 3.0) -> Tuple[List[float], List[int]]:
    """
    Filter outliers using Z-score method.
    
    Args:
        values: List of values to filter
        threshold: Z-score threshold (default 3.0, typical for outliers)
        
    Returns:
        Tuple of (filtered_values, filtered_indices)
    """
    if len(values) < 3:  # Need at least 3 values for Z-score
        return values, list(range(len(values)))
    
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    
    if std_dev == 0:
        return values, list(range(len(values)))
    
    filtered_values = []
    filtered_indices = []
    
    for i, val in enumerate(values):
        z_score = abs((val - mean) / std_dev)
        if z_score <= threshold:
            filtered_values.append(val)
            filtered_indices.append(i)
    
    return filtered_values, filtered_indices


def filter_model_data(
    model_data: Dict,
    filter_method: Optional[str] = None,
    filter_metric: str = "rank_reduction",
    **filter_kwargs
) -> Tuple[Dict, Dict]:
    """
    Filter outliers from model data.
    
    Args:
        model_data: Dictionary containing model data
        filter_method: Filtering method ("iqr", "zscore", or None)
        filter_metric: Metric to filter on ("rank_reduction", "perplexity_increase", or "perplexity_post")
        **filter_kwargs: Additional arguments for filtering (e.g., multiplier, threshold)
        
    Returns:
        Tuple of (filtered_model_data, filter_info)
    """
    if filter_method is None:
        return model_data, {"filtered": False, "method": None, "num_filtered": 0, "total": len(model_data.get("rank_reductions", []))}
    
    original_count = len(model_data.get("rank_reductions", []))
    
    if filter_metric == "rank_reduction":
        values = model_data.get("rank_reductions", [])
    elif filter_metric == "perplexity_increase":
        baseline_ppl = model_data.get("baseline_perplexities", [])
        post_ppl = model_data.get("post_attack_perplexities", [])
        if len(baseline_ppl) != len(post_ppl) or not baseline_ppl:
            return model_data, {"filtered": False, "method": None, "num_filtered": 0, "total": original_count, "reason": "Insufficient perplexity data"}
        values = []
        for b, p in zip(baseline_ppl, post_ppl):
            if b > 0:
                values.append(((p - b) / b) * 100)
            else:
                values.append(0.0)
    elif filter_metric == "perplexity_post":
        values = model_data.get("post_attack_perplexities", [])
    else:
        return model_data, {"filtered": False, "method": None, "num_filtered": 0, "total": original_count, "reason": f"Unknown metric: {filter_metric}"}
    
    if not values:
        return model_data, {"filtered": False, "method": None, "num_filtered": 0, "total": original_count, "reason": "No values to filter"}
    
    # Apply filtering
    if filter_method == "iqr":
        multiplier = filter_kwargs.get("multiplier", 1.5)
        filtered_values, filtered_indices = filter_outliers_iqr(values, multiplier)
    elif filter_method == "zscore":
        threshold = filter_kwargs.get("threshold", 3.0)
        filtered_values, filtered_indices = filter_outliers_zscore(values, threshold)
    else:
        return model_data, {"filtered": False, "method": None, "num_filtered": 0, "total": original_count, "reason": f"Unknown method: {filter_method}"}
    
    num_filtered = original_count - len(filtered_indices)
    
    if num_filtered == 0:
        return model_data, {"filtered": False, "method": filter_method, "num_filtered": 0, "total": original_count}
    
    # Create filtered model data
    filtered_data = {
        "model_name": model_data["model_name"],
        "model_size": model_data["model_size"],
        "rank_reductions": [model_data["rank_reductions"][i] for i in filtered_indices],
        "baseline_ranks": [model_data["baseline_ranks"][i] for i in filtered_indices],
        "post_attack_ranks": [model_data["post_attack_ranks"][i] for i in filtered_indices],
        "baseline_perplexities": [model_data["baseline_perplexities"][i] for i in filtered_indices] if model_data.get("baseline_perplexities") else [],
        "post_attack_perplexities": [model_data["post_attack_perplexities"][i] for i in filtered_indices] if model_data.get("post_attack_perplexities") else [],
        "num_runs": len(filtered_indices),
        "precision": model_data.get("precision"),
        "treatment": model_data.get("treatment"),
        "optimizer": model_data.get("optimizer"),
    }
    
    filter_info = {
        "filtered": True,
        "method": filter_method,
        "metric": filter_metric,
        "num_filtered": num_filtered,
        "total": original_count,
        "filtered_indices": filtered_indices,
        "removed_indices": [i for i in range(original_count) if i not in filtered_indices]
    }
    
    return filtered_data, filter_info


def compute_statistics(values: List[float], use_absolute: bool = True) -> Dict:
    """
    Compute statistics for a list of values.
    
    Args:
        values: List of values
        use_absolute: If True, use absolute values for statistics
        
    Returns:
        Dictionary with mean, std_dev, min, max, count
    """
    if not values:
        return {
            "mean": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }
    
    if use_absolute:
        abs_values = [abs(v) for v in values]
    else:
        abs_values = values
    
    if len(abs_values) == 1:
        return {
            "mean": abs_values[0],
            "std_dev": 0.0,
            "min": abs_values[0],
            "max": abs_values[0],
            "count": 1,
        }
    
    return {
        "mean": statistics.mean(abs_values),
        "std_dev": statistics.stdev(abs_values) if len(abs_values) > 1 else 0.0,
        "min": min(abs_values),
        "max": max(abs_values),
        "count": len(abs_values),
    }


def perform_ttest(group_a: List[float], group_b: List[float]) -> Dict:
    """
    Perform independent t-test between two groups.
    
    Args:
        group_a: First group of values
        group_b: Second group of values
        
    Returns:
        Dictionary with t_statistic, p_value, significant
    """
    if not HAS_SCIPY:
        return {
            "t_statistic": None,
            "p_value": None,
            "significant": None,
            "error": "scipy not available"
        }
    
    try:
        t_stat, p_value = scipy_stats.ttest_ind(group_a, group_b)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "error": None
        }
    except Exception as e:
        return {
            "t_statistic": None,
            "p_value": None,
            "significant": None,
            "error": str(e)
        }


def perform_anova(groups: Dict[str, List[float]]) -> Dict:
    """
    Perform one-way ANOVA for multiple groups.
    
    Args:
        groups: Dictionary mapping group names to lists of values
        
    Returns:
        Dictionary with f_statistic, p_value, significant
    """
    if not HAS_SCIPY:
        return {
            "f_statistic": None,
            "p_value": None,
            "significant": None,
            "error": "scipy not available"
        }
    
    try:
        group_lists = list(groups.values())
        f_stat, p_value = scipy_stats.f_oneway(*group_lists)
        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "error": None
        }
    except Exception as e:
        return {
            "f_statistic": None,
            "p_value": None,
            "significant": None,
            "error": str(e)
        }


def format_pvalue(p_value: Optional[float]) -> str:
    """
    Format p-value with significance markers.
    
    Args:
        p_value: P-value to format
        
    Returns:
        Formatted string with significance markers
    """
    if p_value is None:
        return "N/A"
    
    if p_value < 0.001:
        return f"{p_value:.2e} ***"
    elif p_value < 0.01:
        return f"{p_value:.3f} **"
    elif p_value < 0.05:
        return f"{p_value:.3f} *"
    else:
        return f"{p_value:.3f} ns"


def get_significance_markers_legend() -> Dict[str, str]:
    """
    Get legend explaining significance markers.
    
    Returns:
        Dictionary mapping markers to their meanings
    """
    return {
        "***": "p < 0.001 (highly significant)",
        "**": "p < 0.01 (very significant)",
        "*": "p < 0.05 (significant)",
        "ns": "p >= 0.05 (not significant)"
    }


# Priority-specific plotting functions

def plot_scaling_law(
    all_model_data: List[Dict],
    output_path: Path,
    use_absolute: bool = True,
    title: str = "Scaling Law: Model Size vs Structural Damage"
):
    """
    Plot scaling law curve with error bars (Priority 1).
    
    Args:
        all_model_data: List of model data dictionaries
        output_path: Path to save plot
        use_absolute: Use absolute values for structural damage
        title: Plot title
    """
    if not all_model_data:
        warnings.warn("No data to plot")
        return
    
    # Sort by model size
    sorted_data = sorted(all_model_data, key=lambda x: x["model_size"])
    
    model_sizes = []
    means = []
    std_devs = []
    model_names = []
    
    for data in sorted_data:
        stats = compute_statistics(data["rank_reductions"], use_absolute=use_absolute)
        
        if stats["count"] > 0:
            model_sizes.append(data["model_size"])
            means.append(stats["mean"])
            std_devs.append(stats["std_dev"])
            model_names.append(data["model_name"])
    
    if not model_sizes:
        warnings.warn("No valid data points to plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy arrays for easier manipulation
    model_sizes_arr = np.array(model_sizes)
    means_arr = np.array(means)
    std_devs_arr = np.array(std_devs)
    
    # Set random seed once for reproducibility (for jitter)
    np.random.seed(42)
    
    # Plot with error bars
    ax.errorbar(
        model_sizes_arr,
        means_arr,
        yerr=std_devs_arr,
        fmt='o-',
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label='Mean ± Std Dev',
        color='#2E86AB'
    )
    
    # Add individual data points (optional, for transparency)
    for data in sorted_data:
        stats = compute_statistics(data["rank_reductions"], use_absolute=use_absolute)
        if stats["count"] > 0:
            if use_absolute:
                individual_values = [abs(v) for v in data["rank_reductions"]]
            else:
                individual_values = data["rank_reductions"]
            
            # Scatter individual points with slight jitter
            x_positions = np.full(len(individual_values), data["model_size"])
            x_jitter = x_positions * (1 + np.random.normal(0, 0.02, len(individual_values)))
            ax.scatter(
                x_jitter,
                individual_values,
                alpha=0.3,
                s=30,
                color='#A23B72',
                zorder=0
            )
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Format x-axis labels
    ax.set_xlabel('Model Size (Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Structural Damage (% Rank Reduction)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add model name labels
    for i, (size, mean, name) in enumerate(zip(model_sizes_arr, means_arr, model_names)):
        ax.annotate(
            name.upper(),
            xy=(size, mean),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.7
        )
    
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scaling law plot saved to: {output_path}")
    plt.close()


def plot_treatment_comparison(
    treatment_data: List[Dict],
    output_path: Path,
    title: str = "Placebo Test: Treatment Specificity",
    p_values: Optional[Dict[str, float]] = None
):
    """
    Plot bar chart comparing treatments (Priority 2).
    
    Args:
        treatment_data: List of dicts with treatment stats
        output_path: Path to save plot
        title: Plot title
        p_values: Optional dict mapping treatment pairs to p-values
    """
    if not treatment_data:
        warnings.warn("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    treatments = [d["treatment"] for d in treatment_data]
    means = [d["mean_rank_reduction"] for d in treatment_data]
    std_devs = [d["std_dev"] for d in treatment_data]
    
    # Format treatment names for display
    display_names = {
        "eigen_prion": "Eigen-Prion",
        "gaussian_noise": "Gaussian Noise",
        "random_text": "Random Text"
    }
    display_treatments = [display_names.get(t, t.replace("_", " ").title()) for t in treatments]
    
    x_pos = np.arange(len(treatments))
    bars = ax.bar(x_pos, means, yerr=std_devs, capsize=5, alpha=0.7, color=['#2E86AB', '#A23B72', '#F18F01'])
    
    # Add significance markers if p-values provided
    if p_values:
        max_y = max(m + s for m, s in zip(means, std_devs))
        y_offset = max_y * 0.1
        
        for pair, p_val in p_values.items():
            if p_val < 0.05:
                # Find indices of treatments
                t1, t2 = pair.split(" vs ")
                try:
                    idx1 = treatments.index(t1)
                    idx2 = treatments.index(t2)
                    
                    # Draw bracket
                    x1, x2 = x_pos[idx1], x_pos[idx2]
                    y = max(means[idx1] + std_devs[idx1], means[idx2] + std_devs[idx2]) + y_offset
                    
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset * 0.3, y + y_offset * 0.3, y], 'k-', lw=1)
                    ax.text((x1 + x2) / 2, y + y_offset * 0.4, format_pvalue(p_val), 
                           ha='center', va='bottom', fontsize=9)
                except ValueError:
                    pass
    
    ax.set_xlabel('Treatment Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Rank Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_treatments)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Treatment comparison plot saved to: {output_path}")
    plt.close()


def plot_optimizer_comparison(
    optimizer_data: List[Dict],
    output_path: Path,
    title: str = "Mechanism Test: Optimizer Effect",
    p_value: Optional[float] = None
):
    """
    Plot side-by-side bar chart comparing optimizers (Priority 3).
    
    Args:
        optimizer_data: List of dicts with optimizer stats
        output_path: Path to save plot
        title: Plot title
        p_value: Optional p-value from t-test
    """
    if not optimizer_data:
        warnings.warn("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    optimizers = [d["optimizer"] for d in optimizer_data]
    means = [d["mean_rank_reduction"] for d in optimizer_data]
    std_devs = [d["std_dev"] for d in optimizer_data]
    
    display_names = {
        "adamw": "AdamW",
        "sgd": "SGD"
    }
    display_optimizers = [display_names.get(o, o.upper()) for o in optimizers]
    
    x_pos = np.arange(len(optimizers))
    bars = ax.bar(x_pos, means, yerr=std_devs, capsize=5, alpha=0.7, color=['#2E86AB', '#A23B72'])
    
    # Add p-value annotation
    if p_value is not None:
        max_y = max(m + s for m, s in zip(means, std_devs))
        y_offset = max_y * 0.15
        
        ax.plot([0, 1], [max_y + y_offset, max_y + y_offset], 'k-', lw=1)
        ax.text(0.5, max_y + y_offset * 1.2, f"p = {format_pvalue(p_value)}", 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Rank Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_optimizers)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Optimizer comparison plot saved to: {output_path}")
    plt.close()


def plot_shield_effectiveness(
    precision_data: List[Dict],
    output_path: Path,
    title: str = "Shield Matrix: Quantization Defense",
    p_values: Optional[Dict[str, float]] = None
):
    """
    Plot bar chart showing quantization shield effects (Priority 4).
    
    Args:
        precision_data: List of dicts with precision stats
        output_path: Path to save plot
        title: Plot title
        p_values: Optional dict mapping precision pairs to p-values
    """
    if not precision_data:
        warnings.warn("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    precisions = [d["precision"] for d in precision_data]
    means = [d["mean_rank_reduction"] for d in precision_data]
    std_devs = [d["std_dev"] for d in precision_data]
    
    display_names = {
        "fp16": "FP16",
        "8bit": "8-bit",
        "4bit": "4-bit"
    }
    display_precisions = [display_names.get(p, p.upper()) for p in precisions]
    
    x_pos = np.arange(len(precisions))
    bars = ax.bar(x_pos, means, yerr=std_devs, capsize=5, alpha=0.7, color=['#2E86AB', '#F18F01', '#06A77D'])
    
    # Add significance markers if p-values provided
    if p_values:
        max_y = max(m + s for m, s in zip(means, std_devs))
        y_offset = max_y * 0.1
        
        for pair, p_val in p_values.items():
            if p_val < 0.05:
                p1, p2 = pair.split(" vs ")
                try:
                    idx1 = precisions.index(p1)
                    idx2 = precisions.index(p2)
                    
                    x1, x2 = x_pos[idx1], x_pos[idx2]
                    y = max(means[idx1] + std_devs[idx1], means[idx2] + std_devs[idx2]) + y_offset
                    
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset * 0.3, y + y_offset * 0.3, y], 'k-', lw=1)
                    ax.text((x1 + x2) / 2, y + y_offset * 0.4, format_pvalue(p_val), 
                           ha='center', va='bottom', fontsize=9)
                except ValueError:
                    pass
    
    ax.set_xlabel('Quantization Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Rank Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_precisions)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Shield effectiveness plot saved to: {output_path}")
    plt.close()


# Priority-specific analysis functions

def analyze_priority1_scaling_law(
    results_dir: Path,
    models: List[str],
    output_dir: Path,
    use_absolute: bool = True,
    filter_method: Optional[str] = None,
    filter_metric: str = "rank_reduction",
    filter_iqr_multiplier: float = 1.5,
    filter_zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Analyze scaling law across model sizes (Priority 1).
    
    Args:
        results_dir: Path to results directory
        models: List of model names to analyze
        output_dir: Output directory for results
        use_absolute: Use absolute values for structural damage
        filter_method: Outlier filtering method
        filter_metric: Metric to filter on
        filter_iqr_multiplier: IQR multiplier for filtering
        filter_zscore_threshold: Z-score threshold for filtering
        
    Returns:
        Summary DataFrame
    """
    print("=" * 60)
    print("Priority 1: Scaling Law Analysis")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Models to analyze: {models}")
    print(f"Output directory: {output_dir}")
    print(f"Use absolute values: {use_absolute}")
    if filter_method:
        print(f"Outlier filtering: {filter_method} (metric: {filter_metric})")
    else:
        print("Outlier filtering: None")
    print("=" * 60)
    print()
    
    all_model_data = []
    all_filter_info = []
    
    for model_name in models:
        print(f"Loading data for {model_name}...")
        model_data = load_experiment_data(results_dir, model_name)
        
        if model_data:
            original_count = model_data["num_runs"]
            stats = compute_statistics(model_data["rank_reductions"], use_absolute=use_absolute)
            print(f"  Found {stats['count']} runs (before filtering)")
            print(f"  Mean rank reduction: {stats['mean']:.2f}% ± {stats['std_dev']:.2f}%")
            
            # Apply filtering if requested
            if filter_method:
                filter_kwargs = {}
                if filter_method == "iqr":
                    filter_kwargs["multiplier"] = filter_iqr_multiplier
                elif filter_method == "zscore":
                    filter_kwargs["threshold"] = filter_zscore_threshold
                
                model_data, filter_info = filter_model_data(
                    model_data,
                    filter_method=filter_method,
                    filter_metric=filter_metric,
                    **filter_kwargs
                )
                all_filter_info.append(filter_info)
                
                if filter_info["filtered"]:
                    filtered_count = model_data["num_runs"]
                    print(f"  Filtered: {filter_info['num_filtered']} outlier(s) removed")
                    print(f"  Remaining: {filtered_count} runs")
                    stats_filtered = compute_statistics(model_data["rank_reductions"], use_absolute=use_absolute)
                    print(f"  Mean rank reduction (filtered): {stats_filtered['mean']:.2f}% ± {stats_filtered['std_dev']:.2f}%")
                else:
                    print(f"  No outliers detected")
            else:
                all_filter_info.append({"filtered": False, "method": None, "num_filtered": 0, "total": original_count})
            
            all_model_data.append(model_data)
        else:
            print(f"  Warning: No data found for {model_name}")
            all_filter_info.append({"filtered": False, "method": None, "num_filtered": 0, "total": 0})
        print()
    
    if not all_model_data:
        raise ValueError("No valid data found for any model")
    
    # Generate summary report
    print("Generating summary report...")
    summary_df = generate_summary_report(
        all_model_data,
        output_dir / "scaling_law_summary",
        use_absolute=use_absolute,
        filter_info_list=all_filter_info if filter_method else None
    )
    print()
    
    # Print summary table
    print("=" * 60)
    print("Summary Statistics")
    if filter_method:
        print(f"(After filtering: {filter_method} on {filter_metric})")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    
    # Generate scaling law plot
    print("Generating scaling law plot...")
    plot_path = output_dir / "scaling_law_curve.png"
    plot_scaling_law(
        all_model_data,
        plot_path,
        use_absolute=use_absolute,
        title="Scaling Law: Model Size vs Structural Damage"
    )
    print()
    
    return summary_df


def analyze_priority2_placebo(
    results_dir: Path,
    model: str,
    treatments: List[str],
    output_dir: Path,
    precision: str = "fp16",
) -> pd.DataFrame:
    """
    Compare treatment effects (Priority 2: Placebo Test).
    
    Args:
        results_dir: Path to results directory
        model: Model name to analyze
        treatments: List of treatment types to compare
        output_dir: Output directory for results
        precision: Precision string (default: "fp16")
        
    Returns:
        Summary DataFrame with treatment comparisons
    """
    print("=" * 60)
    print("Priority 2: Placebo Test Analysis")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Treatments: {treatments}")
    print(f"Precision: {precision}")
    print("=" * 60)
    print()
    
    all_treatment_data = []
    treatment_groups = {}
    
    for treatment in treatments:
        print(f"Loading data for {treatment}...")
        data = load_experiment_data(results_dir, model, precision=precision, treatment=treatment)
        
        if data:
            stats = compute_statistics(data["rank_reductions"], use_absolute=True)
            print(f"  Found {stats['count']} runs")
            print(f"  Mean rank reduction: {stats['mean']:.2f}% ± {stats['std_dev']:.2f}%")
            
            # Extract perplexity data
            baseline_perplexities = data.get("baseline_perplexities", [])
            post_attack_perplexities = data.get("post_attack_perplexities", [])
            
            all_treatment_data.append({
                "treatment": treatment,
                "mean_rank_reduction": stats["mean"],
                "std_dev": stats["std_dev"],
                "min": stats["min"],
                "max": stats["max"],
                "n": stats["count"],
                "rank_reductions": data["rank_reductions"],
                "baseline_perplexities": baseline_perplexities,
                "post_attack_perplexities": post_attack_perplexities
            })
            treatment_groups[treatment] = data["rank_reductions"]
        else:
            print(f"  Warning: No data found for {treatment}")
        print()
    
    if len(all_treatment_data) < 2:
        raise ValueError(f"Need at least 2 treatments for comparison, found {len(all_treatment_data)}")
    
    # Perform statistical tests
    print("Performing statistical tests...")
    anova_result = perform_anova(treatment_groups)
    print(f"  ANOVA: F = {anova_result.get('f_statistic', 'N/A'):.3f}, p = {format_pvalue(anova_result.get('p_value'))}")
    
    # Pairwise comparisons
    pairwise_results = {}
    treatment_names = list(treatment_groups.keys())
    for i in range(len(treatment_names)):
        for j in range(i + 1, len(treatment_names)):
            t1, t2 = treatment_names[i], treatment_names[j]
            ttest_result = perform_ttest(treatment_groups[t1], treatment_groups[t2])
            pair_key = f"{t1} vs {t2}"
            pairwise_results[pair_key] = ttest_result.get("p_value")
            print(f"  {pair_key}: t = {ttest_result.get('t_statistic', 'N/A'):.3f}, p = {format_pvalue(ttest_result.get('p_value'))}")
    print()
    
    # Generate summary DataFrame
    summary_rows = []
    for td in all_treatment_data:
        summary_row = {
            "treatment": td["treatment"],
            "mean_rank_reduction_pct": td["mean_rank_reduction"],
            "std_dev_rank_reduction_pct": td["std_dev"],
            "min_rank_reduction_pct": td["min"],
            "max_rank_reduction_pct": td["max"],
            "num_runs": td["n"]
        }
        
        # Add perplexity statistics if available
        baseline_perplexities = td.get("baseline_perplexities", [])
        post_attack_perplexities = td.get("post_attack_perplexities", [])
        
        if baseline_perplexities:
            summary_row["mean_baseline_perplexity"] = statistics.mean(baseline_perplexities)
            summary_row["std_dev_baseline_perplexity"] = statistics.stdev(baseline_perplexities) if len(baseline_perplexities) > 1 else 0.0
            summary_row["min_baseline_perplexity"] = min(baseline_perplexities)
            summary_row["max_baseline_perplexity"] = max(baseline_perplexities)
        
        if post_attack_perplexities:
            summary_row["mean_post_attack_perplexity"] = statistics.mean(post_attack_perplexities)
            summary_row["std_dev_post_attack_perplexity"] = statistics.stdev(post_attack_perplexities) if len(post_attack_perplexities) > 1 else 0.0
            summary_row["min_post_attack_perplexity"] = min(post_attack_perplexities)
            summary_row["max_post_attack_perplexity"] = max(post_attack_perplexities)
        
        # Compute perplexity increase percentage if both are available
        if baseline_perplexities and post_attack_perplexities and len(baseline_perplexities) == len(post_attack_perplexities):
            perplexity_increases = []
            for baseline_ppl, post_ppl in zip(baseline_perplexities, post_attack_perplexities):
                if baseline_ppl > 0:
                    increase_pct = ((post_ppl - baseline_ppl) / baseline_ppl) * 100
                    perplexity_increases.append(increase_pct)
            
            if perplexity_increases:
                summary_row["mean_perplexity_increase_pct"] = statistics.mean(perplexity_increases)
                summary_row["std_dev_perplexity_increase_pct"] = statistics.stdev(perplexity_increases) if len(perplexity_increases) > 1 else 0.0
                summary_row["min_perplexity_increase_pct"] = min(perplexity_increases)
                summary_row["max_perplexity_increase_pct"] = max(perplexity_increases)
        
        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    csv_path = output_dir / "placebo_summary.csv"
    json_path = output_dir / "placebo_summary.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Summary report (CSV) saved to: {csv_path}")
    
    # Convert numpy types to Python native types for JSON serialization
    anova_result_serializable = {
        "f_statistic": float(anova_result.get("f_statistic")) if anova_result.get("f_statistic") is not None else None,
        "p_value": float(anova_result.get("p_value")) if anova_result.get("p_value") is not None else None,
        "significant": bool(anova_result.get("significant")) if anova_result.get("significant") is not None else None,
        "error": anova_result.get("error")
    }
    
    output_data = {
        "summary": summary_rows,
        "statistical_tests": {
            "anova": anova_result_serializable,
            "pairwise": {k: {"p_value": float(v) if v is not None else None} for k, v in pairwise_results.items()}
        },
        "metadata": {
            "significance_markers": get_significance_markers_legend(),
            "note": "Significance markers (***, **, *, ns) are used in console output and plots to indicate statistical significance levels"
        }
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Summary report (JSON) saved to: {json_path}")
    print()
    
    # Generate plot
    print("Generating treatment comparison plot...")
    plot_path = output_dir / "placebo_comparison.png"
    plot_treatment_comparison(all_treatment_data, plot_path, p_values=pairwise_results)
    print()
    
    return summary_df


def analyze_priority3_mechanism(
    results_dir: Path,
    model: str,
    optimizers: List[str],
    output_dir: Path,
    precision: str = "fp16",
) -> pd.DataFrame:
    """
    Compare optimizer effects (Priority 3: Mechanism Test).
    
    Args:
        results_dir: Path to results directory
        model: Model name to analyze
        optimizers: List of optimizers to compare
        output_dir: Output directory for results
        precision: Precision string (default: "fp16")
        
    Returns:
        Summary DataFrame with optimizer comparisons
    """
    print("=" * 60)
    print("Priority 3: Mechanism Test Analysis")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Optimizers: {optimizers}")
    print(f"Precision: {precision}")
    print("=" * 60)
    print()
    
    all_optimizer_data = []
    optimizer_groups = {}
    
    for optimizer in optimizers:
        print(f"Loading data for {optimizer}...")
        data = load_experiment_data(results_dir, model, precision=precision, optimizer=optimizer)
        
        if data:
            stats = compute_statistics(data["rank_reductions"], use_absolute=True)
            print(f"  Found {stats['count']} runs")
            print(f"  Mean rank reduction: {stats['mean']:.2f}% ± {stats['std_dev']:.2f}%")
            
            # Extract perplexity data
            baseline_perplexities = data.get("baseline_perplexities", [])
            post_attack_perplexities = data.get("post_attack_perplexities", [])
            
            all_optimizer_data.append({
                "optimizer": optimizer,
                "mean_rank_reduction": stats["mean"],
                "std_dev": stats["std_dev"],
                "min": stats["min"],
                "max": stats["max"],
                "n": stats["count"],
                "rank_reductions": data["rank_reductions"],
                "baseline_perplexities": baseline_perplexities,
                "post_attack_perplexities": post_attack_perplexities
            })
            optimizer_groups[optimizer] = data["rank_reductions"]
        else:
            print(f"  Warning: No data found for {optimizer}")
        print()
    
    if len(all_optimizer_data) < 2:
        raise ValueError(f"Need at least 2 optimizers for comparison, found {len(all_optimizer_data)}")
    
    # Perform t-test between optimizers
    print("Performing statistical test...")
    optimizer_names = list(optimizer_groups.keys())
    ttest_result = None
    anova_result = None
    p_value = None
    
    if len(optimizer_names) == 2:
        ttest_result = perform_ttest(optimizer_groups[optimizer_names[0]], optimizer_groups[optimizer_names[1]])
        p_value = ttest_result.get("p_value")
        print(f"  t-test: t = {ttest_result.get('t_statistic', 'N/A'):.3f}, p = {format_pvalue(p_value)}")
    else:
        anova_result = perform_anova(optimizer_groups)
        p_value = anova_result.get("p_value")
        print(f"  ANOVA: F = {anova_result.get('f_statistic', 'N/A'):.3f}, p = {format_pvalue(p_value)}")
    print()
    
    # Generate summary DataFrame
    summary_rows = []
    for od in all_optimizer_data:
        summary_row = {
            "optimizer": od["optimizer"],
            "mean_rank_reduction_pct": od["mean_rank_reduction"],
            "std_dev_rank_reduction_pct": od["std_dev"],
            "min_rank_reduction_pct": od["min"],
            "max_rank_reduction_pct": od["max"],
            "num_runs": od["n"]
        }
        
        # Add perplexity statistics if available
        baseline_perplexities = od.get("baseline_perplexities", [])
        post_attack_perplexities = od.get("post_attack_perplexities", [])
        
        if baseline_perplexities:
            summary_row["mean_baseline_perplexity"] = statistics.mean(baseline_perplexities)
            summary_row["std_dev_baseline_perplexity"] = statistics.stdev(baseline_perplexities) if len(baseline_perplexities) > 1 else 0.0
            summary_row["min_baseline_perplexity"] = min(baseline_perplexities)
            summary_row["max_baseline_perplexity"] = max(baseline_perplexities)
        
        if post_attack_perplexities:
            summary_row["mean_post_attack_perplexity"] = statistics.mean(post_attack_perplexities)
            summary_row["std_dev_post_attack_perplexity"] = statistics.stdev(post_attack_perplexities) if len(post_attack_perplexities) > 1 else 0.0
            summary_row["min_post_attack_perplexity"] = min(post_attack_perplexities)
            summary_row["max_post_attack_perplexity"] = max(post_attack_perplexities)
        
        # Compute perplexity increase percentage if both are available
        if baseline_perplexities and post_attack_perplexities and len(baseline_perplexities) == len(post_attack_perplexities):
            perplexity_increases = []
            for baseline_ppl, post_ppl in zip(baseline_perplexities, post_attack_perplexities):
                if baseline_ppl > 0:
                    increase_pct = ((post_ppl - baseline_ppl) / baseline_ppl) * 100
                    perplexity_increases.append(increase_pct)
            
            if perplexity_increases:
                summary_row["mean_perplexity_increase_pct"] = statistics.mean(perplexity_increases)
                summary_row["std_dev_perplexity_increase_pct"] = statistics.stdev(perplexity_increases) if len(perplexity_increases) > 1 else 0.0
                summary_row["min_perplexity_increase_pct"] = min(perplexity_increases)
                summary_row["max_perplexity_increase_pct"] = max(perplexity_increases)
        
        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    csv_path = output_dir / "mechanism_summary.csv"
    json_path = output_dir / "mechanism_summary.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Summary report (CSV) saved to: {csv_path}")
    
    # Prepare statistical test results for JSON serialization
    statistical_test_data = {}
    if ttest_result:
        statistical_test_data = {
            "test_type": "t-test",
            "t_statistic": float(ttest_result.get("t_statistic")) if ttest_result.get("t_statistic") is not None else None,
            "p_value": float(ttest_result.get("p_value")) if ttest_result.get("p_value") is not None else None,
            "significant": bool(ttest_result.get("p_value", 1.0) < 0.05) if ttest_result.get("p_value") is not None else None
        }
    elif anova_result:
        statistical_test_data = {
            "test_type": "anova",
            "f_statistic": float(anova_result.get("f_statistic")) if anova_result.get("f_statistic") is not None else None,
            "p_value": float(anova_result.get("p_value")) if anova_result.get("p_value") is not None else None,
            "significant": bool(anova_result.get("p_value", 1.0) < 0.05) if anova_result.get("p_value") is not None else None,
            "error": anova_result.get("error")
        }
    else:
        statistical_test_data = {
            "test_type": None,
            "p_value": None,
            "significant": None
        }
    
    output_data = {
        "summary": summary_rows,
        "statistical_test": statistical_test_data,
        "metadata": {
            "significance_markers": get_significance_markers_legend(),
            "note": "Significance markers (***, **, *, ns) are used in console output and plots to indicate statistical significance levels"
        }
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Summary report (JSON) saved to: {json_path}")
    print()
    
    # Generate plot
    print("Generating optimizer comparison plot...")
    plot_path = output_dir / "mechanism_comparison.png"
    plot_optimizer_comparison(all_optimizer_data, plot_path, p_value=p_value)
    print()
    
    return summary_df


def analyze_priority4_shield(
    results_dir: Path,
    model: str,
    precisions: List[str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compare quantization shield effects (Priority 4: Shield Matrix).
    
    Args:
        results_dir: Path to results directory
        model: Model name to analyze
        precisions: List of precision levels to compare
        output_dir: Output directory for results
        
    Returns:
        Summary DataFrame with precision comparisons
    """
    print("=" * 60)
    print("Priority 4: Shield Matrix Analysis")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Precisions: {precisions}")
    print("=" * 60)
    print()
    
    all_precision_data = []
    precision_groups = {}
    
    for precision in precisions:
        print(f"Loading data for {precision}...")
        data = load_experiment_data(results_dir, model, precision=precision)
        
        if data:
            stats = compute_statistics(data["rank_reductions"], use_absolute=True)
            print(f"  Found {stats['count']} runs")
            print(f"  Mean rank reduction: {stats['mean']:.2f}% ± {stats['std_dev']:.2f}%")
            
            all_precision_data.append({
                "precision": precision,
                "mean_rank_reduction": stats["mean"],
                "std_dev": stats["std_dev"],
                "min": stats["min"],
                "max": stats["max"],
                "n": stats["count"],
                "rank_reductions": data["rank_reductions"]
            })
            precision_groups[precision] = data["rank_reductions"]
        else:
            print(f"  Warning: No data found for {precision}")
        print()
    
    if len(all_precision_data) < 2:
        raise ValueError(f"Need at least 2 precision levels for comparison, found {len(all_precision_data)}")
    
    # Perform statistical tests
    print("Performing statistical tests...")
    anova_result = perform_anova(precision_groups)
    print(f"  ANOVA: F = {anova_result.get('f_statistic', 'N/A'):.3f}, p = {format_pvalue(anova_result.get('p_value'))}")
    
    # Pairwise comparisons
    pairwise_results = {}
    precision_names = list(precision_groups.keys())
    for i in range(len(precision_names)):
        for j in range(i + 1, len(precision_names)):
            p1, p2 = precision_names[i], precision_names[j]
            ttest_result = perform_ttest(precision_groups[p1], precision_groups[p2])
            pair_key = f"{p1} vs {p2}"
            pairwise_results[pair_key] = ttest_result.get("p_value")
            print(f"  {pair_key}: t = {ttest_result.get('t_statistic', 'N/A'):.3f}, p = {format_pvalue(ttest_result.get('p_value'))}")
    print()
    
    # Generate summary DataFrame
    summary_rows = []
    for pd_data in all_precision_data:
        summary_rows.append({
            "precision": pd_data["precision"],
            "mean_rank_reduction_pct": pd_data["mean_rank_reduction"],
            "std_dev_rank_reduction_pct": pd_data["std_dev"],
            "min_rank_reduction_pct": pd_data["min"],
            "max_rank_reduction_pct": pd_data["max"],
            "num_runs": pd_data["n"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    csv_path = output_dir / "shield_summary.csv"
    json_path = output_dir / "shield_summary.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Summary report (CSV) saved to: {csv_path}")
    
    # Convert numpy types to Python native types for JSON serialization
    anova_result_serializable = {
        "f_statistic": float(anova_result.get("f_statistic")) if anova_result.get("f_statistic") is not None else None,
        "p_value": float(anova_result.get("p_value")) if anova_result.get("p_value") is not None else None,
        "significant": bool(anova_result.get("significant")) if anova_result.get("significant") is not None else None,
        "error": anova_result.get("error")
    }
    
    output_data = {
        "summary": summary_rows,
        "statistical_tests": {
            "anova": anova_result_serializable,
            "pairwise": {k: {"p_value": float(v) if v is not None else None} for k, v in pairwise_results.items()}
        },
        "metadata": {
            "significance_markers": get_significance_markers_legend(),
            "note": "Significance markers (***, **, *, ns) are used in console output and plots to indicate statistical significance levels"
        }
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Summary report (JSON) saved to: {json_path}")
    print()
    
    # Generate plot
    print("Generating shield effectiveness plot...")
    plot_path = output_dir / "shield_matrix.png"
    plot_shield_effectiveness(all_precision_data, plot_path, p_values=pairwise_results)
    print()
    
    return summary_df


def generate_summary_report(
    all_model_data: List[Dict],
    output_path: Path,
    use_absolute: bool = True,
    filter_info_list: Optional[List[Dict]] = None
):
    """
    Generate summary report as CSV and JSON.
    
    Args:
        all_model_data: List of model data dictionaries
        output_path: Base path for output files (will create .csv and .json)
        use_absolute: Use absolute values for structural damage
        filter_info_list: Optional list of filter information dictionaries
    """
    if not all_model_data:
        warnings.warn("No data to generate report")
        return pd.DataFrame()
    
    # Sort by model size
    sorted_data = sorted(all_model_data, key=lambda x: x["model_size"])
    
    # Prepare summary data
    summary_rows = []
    
    for data in sorted_data:
        stats = compute_statistics(data["rank_reductions"], use_absolute=use_absolute)
        
        # Compute perplexity statistics
        baseline_perplexities = data.get("baseline_perplexities", [])
        post_attack_perplexities = data.get("post_attack_perplexities", [])
        
        summary_row = {
            "model_name": data["model_name"],
            "model_size": data["model_size"],
            "model_size_millions": data["model_size"] / 1e6,
            "num_runs": stats["count"],
            "mean_rank_reduction_pct": stats["mean"],
            "std_dev_rank_reduction_pct": stats["std_dev"],
            "min_rank_reduction_pct": stats["min"],
            "max_rank_reduction_pct": stats["max"],
            "mean_baseline_rank": statistics.mean(data["baseline_ranks"]) if data["baseline_ranks"] else 0.0,
            "mean_post_attack_rank": statistics.mean(data["post_attack_ranks"]) if data["post_attack_ranks"] else 0.0,
        }
        
        # Add perplexity statistics if available
        if baseline_perplexities:
            summary_row["mean_baseline_perplexity"] = statistics.mean(baseline_perplexities)
            summary_row["std_dev_baseline_perplexity"] = statistics.stdev(baseline_perplexities) if len(baseline_perplexities) > 1 else 0.0
            summary_row["min_baseline_perplexity"] = min(baseline_perplexities)
            summary_row["max_baseline_perplexity"] = max(baseline_perplexities)
        
        if post_attack_perplexities:
            summary_row["mean_post_attack_perplexity"] = statistics.mean(post_attack_perplexities)
            summary_row["std_dev_post_attack_perplexity"] = statistics.stdev(post_attack_perplexities) if len(post_attack_perplexities) > 1 else 0.0
            summary_row["min_post_attack_perplexity"] = min(post_attack_perplexities)
            summary_row["max_post_attack_perplexity"] = max(post_attack_perplexities)
        
        # Compute perplexity increase percentage if both are available
        if baseline_perplexities and post_attack_perplexities and len(baseline_perplexities) == len(post_attack_perplexities):
            perplexity_increases = []
            for baseline_ppl, post_ppl in zip(baseline_perplexities, post_attack_perplexities):
                if baseline_ppl > 0:
                    increase_pct = ((post_ppl - baseline_ppl) / baseline_ppl) * 100
                    perplexity_increases.append(increase_pct)
            
            if perplexity_increases:
                summary_row["mean_perplexity_increase_pct"] = statistics.mean(perplexity_increases)
                summary_row["std_dev_perplexity_increase_pct"] = statistics.stdev(perplexity_increases) if len(perplexity_increases) > 1 else 0.0
                summary_row["min_perplexity_increase_pct"] = min(perplexity_increases)
                summary_row["max_perplexity_increase_pct"] = max(perplexity_increases)
        
        summary_rows.append(summary_row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_rows)
    
    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Summary report (CSV) saved to: {csv_path}")
    
    # Save JSON
    json_path = output_path.with_suffix('.json')
    
    # If filtering was applied, include metadata; otherwise keep original format for backward compatibility
    if filter_info_list and any(info.get("filtered", False) for info in filter_info_list):
        output_data = {
            "summary": summary_rows,
            "filtering_info": filter_info_list,
            "filtering_applied": True
        }
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    else:
        # Backward compatible: just output the summary array
        with open(json_path, 'w') as f:
            json.dump(summary_rows, f, indent=2)
    
    print(f"Summary report (JSON) saved to: {json_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment data across all priorities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Priority 1: Scaling Law
  %(prog)s 1 --results-dir ./experiments/results/ --models 70m 160m 410m 1b --output-dir ./results/priority1/
  
  # Priority 2: Placebo Test
  %(prog)s 2 --results-dir ./experiments/results/ --model 410m --treatments eigen_prion gaussian_noise random_text --output-dir ./results/priority2/
  
  # Priority 3: Mechanism Test
  %(prog)s 3 --results-dir ./experiments/results/ --model 410m --optimizers adamw sgd --output-dir ./results/priority3/
  
  # Priority 4: Shield Matrix
  %(prog)s 4 --results-dir ./experiments/results/ --model 1.4b --precisions fp16 8bit 4bit --output-dir ./results/priority4/
        """
    )
    
    parser.add_argument(
        'priority',
        type=int,
        choices=[1, 2, 3, 4],
        help='Priority type: 1=Scaling Law, 2=Placebo Test, 3=Mechanism Test, 4=Shield Matrix'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./experiments/results/',
        help='Path to results directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for plots and reports'
    )
    
    # Priority 1 specific arguments
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='List of models to analyze (Priority 1 only)'
    )
    parser.add_argument(
        '--use-absolute',
        action='store_true',
        default=True,
        help='Use absolute value for structural damage (Priority 1, default: True)'
    )
    parser.add_argument(
        '--no-absolute',
        dest='use_absolute',
        action='store_false',
        help='Use signed values for structural damage (Priority 1)'
    )
    parser.add_argument(
        '--filter-method',
        type=str,
        choices=['iqr', 'zscore'],
        default=None,
        help='Outlier filtering method (Priority 1)'
    )
    parser.add_argument(
        '--filter-metric',
        type=str,
        choices=['rank_reduction', 'perplexity_increase', 'perplexity_post'],
        default='rank_reduction',
        help='Metric to use for filtering outliers (Priority 1)'
    )
    parser.add_argument(
        '--filter-iqr-multiplier',
        type=float,
        default=1.5,
        help='IQR multiplier for outlier detection (Priority 1, default: 1.5)'
    )
    parser.add_argument(
        '--filter-zscore-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (Priority 1, default: 3.0)'
    )
    
    # Priority 2 specific arguments
    parser.add_argument(
        '--model',
        type=str,
        help='Model name to analyze (Priorities 2, 3, 4)'
    )
    parser.add_argument(
        '--treatments',
        type=str,
        nargs='+',
        default=['eigen_prion', 'gaussian_noise', 'random_text'],
        help='Treatment types to compare (Priority 2)'
    )
    
    # Priority 3 specific arguments
    parser.add_argument(
        '--optimizers',
        type=str,
        nargs='+',
        default=['adamw', 'sgd'],
        help='Optimizers to compare (Priority 3)'
    )
    
    # Priority 4 specific arguments
    parser.add_argument(
        '--precisions',
        type=str,
        nargs='+',
        default=['fp16', '8bit', '4bit'],
        help='Precision levels to compare (Priority 4)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    # Check if results_dir is a direct subdirectory of output_dir
    # This ensures proper separation between raw experiment data and analysis results
    try:
        results_dir_relative = results_dir.relative_to(output_dir)
        # Check if results_dir is exactly one level deep under output_dir
        if len(results_dir_relative.parts) == 1:
            print("=" * 60)
            print("WARNING: Directory Structure Issue Detected")
            print("=" * 60)
            print(f"Results directory: {results_dir}")
            print(f"Output directory: {output_dir}")
            print(f"\nThe results directory is a direct subdirectory of the output directory.")
            print("This mixes raw experiment data with analysis results, which is not recommended.")
            print("\n建议 (Recommendation):")
            print("在后处理阶段，将分析时用到的'原始的实验数据'提前移动到数据分析目录的子目录下。")
            print("(In post-processing, move the 'raw experiment data' used for analysis")
            print(" to a subdirectory under the data analysis directory in advance.)")
            print("\n例如 (Example):")
            print(f"  mkdir -p {output_dir}/raw_data/")
            print(f"  mv {results_dir}/* {output_dir}/raw_data/")
            print(f"  然后使用: --results-dir {output_dir}/raw_data/")
            print("=" * 60)
            print()
    except ValueError:
        # results_dir is not under output_dir, which is fine
        pass
    
    # Dispatch based on priority
    if args.priority == 1:
        if not args.models:
            parser.error("--models is required for Priority 1")
        analyze_priority1_scaling_law(
            results_dir,
            args.models,
            output_dir,
            use_absolute=args.use_absolute,
            filter_method=args.filter_method,
            filter_metric=args.filter_metric,
            filter_iqr_multiplier=args.filter_iqr_multiplier,
            filter_zscore_threshold=args.filter_zscore_threshold,
        )
    elif args.priority == 2:
        if not args.model:
            parser.error("--model is required for Priority 2")
        analyze_priority2_placebo(
            results_dir,
            args.model,
            args.treatments,
            output_dir,
        )
    elif args.priority == 3:
        if not args.model:
            parser.error("--model is required for Priority 3")
        analyze_priority3_mechanism(
            results_dir,
            args.model,
            args.optimizers,
            output_dir,
        )
    elif args.priority == 4:
        if not args.model:
            parser.error("--model is required for Priority 4")
        analyze_priority4_shield(
            results_dir,
            args.model,
            args.precisions,
            output_dir,
        )
    
    print("=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
