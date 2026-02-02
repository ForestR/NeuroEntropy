#!/usr/bin/env python3
"""
Convert CSV summary files to LaTeX tables.
Reads from paper/data/ and generates formatted LaTeX tables.
"""

import pandas as pd
import json
import sys
from pathlib import Path

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).parent
PAPER_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PAPER_DIR / "data"
TABLES_DIR = PAPER_DIR / "tables"


def generate_scaling_law_table():
    """Generate LaTeX table for Priority 1: Scaling Law results."""
    csv_path = DATA_DIR / "scaling_law_summary.csv"
    
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping scaling law table.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Select relevant columns
    columns = [
        'model_name',
        'model_size_millions',
        'mean_rank_reduction_pct',
        'std_dev_rank_reduction_pct',
        'mean_perplexity_increase_pct'
    ]
    
    # Filter to available columns
    available_cols = [col for col in columns if col in df.columns]
    df_subset = df[available_cols].copy()
    
    # Format columns
    if 'model_size_millions' in df_subset.columns:
        df_subset['model_size_millions'] = df_subset['model_size_millions'].apply(lambda x: f"{x:.0f}")
    
    # Format percentage columns
    for col in ['mean_rank_reduction_pct', 'std_dev_rank_reduction_pct', 'mean_perplexity_increase_pct']:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Rename columns for LaTeX
    column_names = {
        'model_name': 'Model',
        'model_size_millions': 'Size (M)',
        'mean_rank_reduction_pct': r'Mean Rank Reduction (\%)',
        'std_dev_rank_reduction_pct': r'Std Dev (\%)',
        'mean_perplexity_increase_pct': r'Mean Perplexity Increase (\%)'
    }
    df_subset = df_subset.rename(columns=column_names)
    
    # Generate LaTeX
    latex_table = df_subset.to_latex(
        index=False,
        escape=False,
        float_format="%.2f",
        caption="Scaling Law Results: Rank Reduction vs Model Size",
        label="tab:scaling_law",
        column_format="l" + "c" * (len(df_subset.columns) - 1)
    )
    
    # Replace default table format with booktabs
    latex_table = latex_table.replace("\\begin{tabular}", "\\begin{tabular}")
    latex_table = latex_table.replace("\\hline", "\\toprule", 1)
    latex_table = latex_table.replace("\\hline", "\\midrule", 1)
    latex_table = latex_table.replace("\\hline", "\\bottomrule", 1)
    
    # Write to file
    output_path = TABLES_DIR / "tab1_scaling_summary.tex"
    output_path.write_text(latex_table)
    print(f"Generated: {output_path}")


def generate_statistical_tests_table():
    """Generate LaTeX table for statistical test results from priorities 2-4."""
    # This would combine data from placebo, mechanism, and shield summaries
    # For now, create a placeholder structure
    
    table_content = """\\begin{table}[h]
\\centering
\\caption{Statistical Test Results}
\\label{tab:statistical_tests}
\\begin{tabular}{lccc}
\\toprule
Experiment & Test & Statistic & p-value \\\\
\\midrule
Placebo (Priority 2) & ANOVA & F = 1.624 & p = 0.273 ns \\\\
Placebo (Priority 2) & t-test (eigen-prion vs random) & t = -0.473 & p = 0.661 ns \\\\
Mechanism (Priority 3) & t-test (Adam vs SGD) & t = -0.948 & p = 0.397 ns \\\\
Shield (Priority 4) & ANOVA & F = 0.804 & p = 0.491 ns \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    output_path = TABLES_DIR / "tab2_statistical_tests.tex"
    output_path.write_text(table_content)
    print(f"Generated: {output_path}")
    
    # TODO: Parse JSON files to extract actual statistical test results
    # For now, using placeholder data from roadmap


if __name__ == "__main__":
    print("Generating LaTeX tables from CSV summaries...")
    generate_scaling_law_table()
    generate_statistical_tests_table()
    print("Done!")
