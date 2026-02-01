#!/usr/bin/env python3
"""
Priority 1 Data Analysis Script

Analyzes completed Priority 1 experiments and generates scaling law visualization
(Model Size vs Structural Damage) with error bars.

Usage:
    python experiments/01_pythia_160m/analyze_priority1.py \
        --results-dir ./experiments/results/ \
        --models 70m 160m 410m 1b \
        --output-dir ./experiments/results/scaling_law_analysis/
    
    # With outlier filtering:
    python experiments/01_pythia_160m/analyze_priority1.py \
        --results-dir ./experiments/results/ \
        --models 70m 160m 410m 1b \
        --output-dir ./experiments/results/scaling_law_analysis/ \
        --filter-method iqr \
        --filter-metric perplexity_increase \
        --filter-iqr-multiplier 1.5
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


def load_model_data(results_dir: Path, model_name: str) -> Optional[Dict]:
    """
    Load experiment data for a single model.
    
    Args:
        results_dir: Path to results directory
        model_name: Model identifier (e.g., "70m", "160m")
        
    Returns:
        Dictionary with model_size, rank_reductions, baseline_ranks, post_attack_ranks
        Returns None if no data found
    """
    # Try to find aggregate CSV file first
    csv_pattern = f"aggregate_summary_pythia_{model_name}_pythia-{model_name}_fp16.csv"
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
                run_dirs = sorted(results_dir.glob(f"pythia_{model_name}_pythia-{model_name}_run*_fp16"))
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
            }
        except Exception as e:
            warnings.warn(f"Error loading CSV {csv_file}: {e}")
            return None
    
    # Fallback: try to parse individual run directories
    run_dirs = sorted(results_dir.glob(f"pythia_{model_name}_pythia-{model_name}_run*_fp16"))
    
    if not run_dirs:
        warnings.warn(f"No data found for model {model_name}")
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


def plot_scaling_law(
    all_model_data: List[Dict],
    output_path: Path,
    use_absolute: bool = True,
    title: str = "Scaling Law: Model Size vs Structural Damage"
):
    """
    Plot scaling law curve with error bars.
    
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
        return
    
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
        description="Analyze Priority 1 experiment data and generate scaling law visualization"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./experiments/results/',
        help='Path to results directory'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['70m', '160m', '410m', '1b'],
        help='List of models to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments/results/scaling_law_analysis/',
        help='Output directory for plots and reports'
    )
    parser.add_argument(
        '--use-absolute',
        action='store_true',
        default=True,
        help='Use absolute value for structural damage (default: True)'
    )
    parser.add_argument(
        '--no-absolute',
        dest='use_absolute',
        action='store_false',
        help='Use signed values for structural damage'
    )
    parser.add_argument(
        '--filter-method',
        type=str,
        choices=['iqr', 'zscore'],
        default=None,
        help='Outlier filtering method: "iqr" (Interquartile Range) or "zscore" (Z-score)'
    )
    parser.add_argument(
        '--filter-metric',
        type=str,
        choices=['rank_reduction', 'perplexity_increase', 'perplexity_post'],
        default='rank_reduction',
        help='Metric to use for filtering outliers (default: rank_reduction)'
    )
    parser.add_argument(
        '--filter-iqr-multiplier',
        type=float,
        default=1.5,
        help='IQR multiplier for outlier detection (default: 1.5)'
    )
    parser.add_argument(
        '--filter-zscore-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    print("=" * 60)
    print("Priority 1 Data Analysis")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Models to analyze: {args.models}")
    print(f"Output directory: {output_dir}")
    print(f"Use absolute values: {args.use_absolute}")
    if args.filter_method:
        print(f"Outlier filtering: {args.filter_method} (metric: {args.filter_metric})")
        if args.filter_method == "iqr":
            print(f"  IQR multiplier: {args.filter_iqr_multiplier}")
        elif args.filter_method == "zscore":
            print(f"  Z-score threshold: {args.filter_zscore_threshold}")
    else:
        print("Outlier filtering: None")
    print("=" * 60)
    print()
    
    # Load data for each model
    all_model_data = []
    all_filter_info = []
    
    for model_name in args.models:
        print(f"Loading data for {model_name}...")
        model_data = load_model_data(results_dir, model_name)
        
        if model_data:
            original_count = model_data["num_runs"]
            stats = compute_statistics(model_data["rank_reductions"], use_absolute=args.use_absolute)
            print(f"  Found {stats['count']} runs (before filtering)")
            print(f"  Mean rank reduction: {stats['mean']:.2f}% ± {stats['std_dev']:.2f}%")
            
            # Apply filtering if requested
            if args.filter_method:
                filter_kwargs = {}
                if args.filter_method == "iqr":
                    filter_kwargs["multiplier"] = args.filter_iqr_multiplier
                elif args.filter_method == "zscore":
                    filter_kwargs["threshold"] = args.filter_zscore_threshold
                
                model_data, filter_info = filter_model_data(
                    model_data,
                    filter_method=args.filter_method,
                    filter_metric=args.filter_metric,
                    **filter_kwargs
                )
                all_filter_info.append(filter_info)
                
                if filter_info["filtered"]:
                    filtered_count = model_data["num_runs"]
                    print(f"  Filtered: {filter_info['num_filtered']} outlier(s) removed ({filter_info['num_filtered']}/{original_count} runs)")
                    print(f"  Remaining: {filtered_count} runs")
                    if filter_info.get("removed_indices"):
                        print(f"  Removed run indices: {filter_info['removed_indices']}")
                    stats_filtered = compute_statistics(model_data["rank_reductions"], use_absolute=args.use_absolute)
                    print(f"  Mean rank reduction (filtered): {stats_filtered['mean']:.2f}% ± {stats_filtered['std_dev']:.2f}%")
                else:
                    print(f"  No outliers detected with {args.filter_method} method")
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
        use_absolute=args.use_absolute,
        filter_info_list=all_filter_info if args.filter_method else None
    )
    print()
    
    # Add filtering information to summary if filtering was applied
    if args.filter_method:
        filter_summary = []
        for i, (model_name, filter_info) in enumerate(zip(args.models, all_filter_info)):
            if filter_info.get("filtered"):
                filter_summary.append({
                    "model": model_name,
                    "filter_method": filter_info["method"],
                    "filter_metric": filter_info.get("metric", "N/A"),
                    "total_runs": filter_info["total"],
                    "filtered_runs": filter_info["num_filtered"],
                    "remaining_runs": filter_info["total"] - filter_info["num_filtered"],
                    "removed_indices": filter_info.get("removed_indices", [])
                })
        
        if filter_summary:
            print("=" * 60)
            print("Outlier Filtering Summary")
            print("=" * 60)
            for info in filter_summary:
                print(f"Model: {info['model']}")
                print(f"  Method: {info['filter_method']} (metric: {info['filter_metric']})")
                print(f"  Total runs: {info['total_runs']}")
                print(f"  Filtered: {info['filtered_runs']} outlier(s)")
                print(f"  Remaining: {info['remaining_runs']} runs")
                if info['removed_indices']:
                    print(f"  Removed run indices: {info['removed_indices']}")
                print()
    
    # Print summary table
    print("=" * 60)
    print("Summary Statistics")
    if args.filter_method:
        print(f"(After filtering: {args.filter_method} on {args.filter_metric})")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()
    
    # Generate scaling law plot
    print("Generating scaling law plot...")
    plot_path = output_dir / "scaling_law_curve.png"
    plot_scaling_law(
        all_model_data,
        plot_path,
        use_absolute=args.use_absolute,
        title="Scaling Law: Model Size vs Structural Damage"
    )
    print()
    
    print("=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
