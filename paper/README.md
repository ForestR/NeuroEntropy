# LaTeX Paper: Inverse Scaling of Structural Robustness

This directory contains the LaTeX source for the manuscript: **"Inverse Scaling of Structural Robustness: Spectral Collapse Induced by Adaptive Optimization in Large Language Models"**.

## Directory Structure

```
paper/
├── main.tex                     # Main document (compile this)
├── sections/                    # Modular content sections
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_theory.tex
│   ├── 03_methods.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   ├── 06_conclusion.tex
│   └── appendix/               # Supplementary materials
├── figures/                     # Publication-ready figures (symlinks)
├── tables/                      # LaTeX table files
│   └── scripts/                # Table generation scripts
├── data/                        # Data files (symlinks to experiments/results/)
├── bibliography/                # BibTeX references
├── style/                       # LaTeX styling and macros
└── scripts/                     # Build automation
```

## Prerequisites

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: `amsmath`, `graphicx`, `booktabs`, `hyperref`, `natbib`
- Python 3 (for table generation scripts)
- `pandas` Python package (for CSV to LaTeX conversion)

## Building the Paper

### Quick Compilation

```bash
cd paper
./scripts/compile.sh
```

This will:
1. Run `pdflatex` three times (for cross-references)
2. Run `bibtex` for bibliography
3. Generate `main.pdf`

### Manual Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Cleaning Auxiliary Files

```bash
./scripts/clean.sh
```

## Data Dependencies

The paper uses symlinks to experimental results in `../experiments/results/`:

- **Priority 1 (Scaling Law)**: `priority1_scaling_law/scaling_law_summary.{json,csv}`
- **Priority 2 (Placebo)**: `priority2_placebo/placebo_summary.{json,csv}`
- **Priority 3 (Mechanism)**: `priority3_mechanism/mechanism_summary.{json,csv}`
- **Priority 4 (Shield)**: `priority4_shield/shield_summary.{json,csv}`

Figures are also symlinked from the results directories.

## Generating Tables

To regenerate LaTeX tables from CSV summaries:

```bash
python tables/scripts/csv_to_latex.py
```

This will generate:
- `tables/tab1_scaling_summary.tex` - Scaling law results
- `tables/tab2_statistical_tests.tex` - Statistical test results

## Word Count

Approximate word count:

```bash
./scripts/word_count.sh
```

For accurate counts, use `texcount` or count manually in the compiled PDF.

## Figure References

The paper includes four main figures:

1. **Figure 1**: Scaling Law (`fig1_scaling_law.png`) - Inverse scaling law showing 1B collapse
2. **Figure 2**: Quantization Shield (`fig2_shield_matrix.png`) - 8-bit defense effectiveness
3. **Figure 3**: Placebo Test (`fig3_placebo_comparison.png`) - Attack specificity
4. **Figure 4**: Mechanism Test (`fig4_mechanism_comparison.png`) - Adam vs SGD

All figures are symlinked from `../experiments/results/`.

## Content Overview

Based on the roadmap, the paper covers:

- **Abstract**: Low-Rank Hypothesis, Metabolic Attack mechanism, 1B collapse vs Quantization immunity
- **Introduction**: Energy vs Structure duality
- **Theory**: Adam Amplification + Norm Coupling mathematics
- **Methods**: Experimental setup across 4 priorities
- **Results**: 
  - The Curve: 70M → 1B progression (inverse scaling law)
  - The Shield: FP16 vs 8-bit (70x improvement)
  - Placebo and Mechanism tests
- **Discussion**: AI Safety implications, DeepSeek mHC, Quantization defense
- **Appendices**: Extended analysis for each priority

## Notes

- Symlinks are used during development to maintain single source of truth
- For arXiv submission, consider copying files instead of using symlinks
- The document class is `article` - can be adapted to venue-specific formats (NeurIPS, ICML, etc.)
- Custom commands are defined in `style/custom_commands.tex` for consistent notation

## Troubleshooting

**Missing figures**: Ensure symlinks are valid (`ls -la figures/`)

**Bibliography errors**: Check that `bibliography/references.bib` exists and contains entries

**Compilation errors**: Check `main.log` for detailed error messages

**Table generation fails**: Ensure Python dependencies are installed (`pip install pandas`)
