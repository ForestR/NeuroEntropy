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
            
            return {
                "model_name": model_name,
                "model_size": model_size,
                "rank_reductions": rank_reductions,
                "baseline_ranks": baseline_ranks,
                "post_attack_ranks": post_attack_ranks,
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
        "num_runs": len(rank_reductions),
    }


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
    use_absolute: bool = True
):
    """
    Generate summary report as CSV and JSON.
    
    Args:
        all_model_data: List of model data dictionaries
        output_path: Base path for output files (will create .csv and .json)
        use_absolute: Use absolute values for structural damage
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
        
        summary_rows.append({
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
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_rows)
    
    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Summary report (CSV) saved to: {csv_path}")
    
    # Save JSON
    json_path = output_path.with_suffix('.json')
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
    print("=" * 60)
    print()
    
    # Load data for each model
    all_model_data = []
    
    for model_name in args.models:
        print(f"Loading data for {model_name}...")
        model_data = load_model_data(results_dir, model_name)
        
        if model_data:
            stats = compute_statistics(model_data["rank_reductions"], use_absolute=args.use_absolute)
            print(f"  Found {stats['count']} runs")
            print(f"  Mean rank reduction: {stats['mean']:.2f}% ± {stats['std_dev']:.2f}%")
            all_model_data.append(model_data)
        else:
            print(f"  Warning: No data found for {model_name}")
        print()
    
    if not all_model_data:
        raise ValueError("No valid data found for any model")
    
    # Generate summary report
    print("Generating summary report...")
    summary_df = generate_summary_report(
        all_model_data,
        output_dir / "scaling_law_summary",
        use_absolute=args.use_absolute
    )
    print()
    
    # Print summary table
    print("=" * 60)
    print("Summary Statistics")
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
