# Pythia Experiments

Experiments demonstrating metabolic attacks (Eigen-Prion) on Pythia models of various sizes.

## Status

🚧 **In Progress**

## Setup

- Models: EleutherAI/pythia-{70m,160m,410m,1b,1.4b,2.8b}
- Hardware: Single GPU (RTX 4090, 24GB)
- Framework: PyTorch + Transformers

**Environment:**

```bash
conda activate neuroentropy
```

## Running experiments

### Single Run

Basic single run (e.g. 160M FP16):

```bash
conda activate neuroentropy
python experiments/01_pythia_160m/run_experiment.py --model 160m --quantization fp16
```

### Priority 1: Scaling Law Experiments

To generate the scaling law curve (Model Size vs. Structural Damage), run **5 repetitions** for each model with **FP16 precision** and **Full Fine-Tuning** mode:

**Models:** 70m, 160m, 410m, 1b, 1.4b, 2.8b  
**Total:** 6 models × 5 runs = 30 experiments

**Command template for each model:**

```bash
conda activate neuroentropy
python experiments/01_pythia_160m/run_experiment.py \
    --model <MODEL> \
    --quantization fp16 \
    --force-fft \
    --num-runs 5 \
    --seed 42 \
    --output-dir ./experiments/results/ \
    --skip-control \
    --verbosity normal
```

**Example: Run 5 repetitions for Pythia-70M:**

```bash
python experiments/01_pythia_160m/run_experiment.py \
    --model 70m \
    --quantization fp16 \
    --force-fft \
    --num-runs 5 \
    --seed 42 \
    --output-dir ./experiments/results/ \
    --skip-control \
    --verbosity normal
```

**Note on seed behavior:** When using `--num-runs N --seed BASE_SEED`, each run will automatically use a different seed:
- Run 1: `BASE_SEED + 1`
- Run 2: `BASE_SEED + 2`
- Run 3: `BASE_SEED + 3`
- ... and so on

This ensures each repetition uses a different random seed for proper statistical analysis.

### ⚠️ Memory Warning: OOM Risk for Large Models

**Important:** Models **1.4B** and **2.8B** may encounter **Out-of-Memory (OOM)** errors on 24GB GPUs (e.g., RTX 4090).

**Known Issues:**
- **1.4B model:** May OOM during attack loop phase (even after optimizations)
- **2.8B model:** Likely to OOM due to activation memory requirements

**Recommendations:**
- For **1B and smaller models:** Should run successfully on 24GB GPU
- For **1.4B/2.8B models:** 
  - Consider using larger GPU memory (≥32GB) if available
  - Or skip these models if OOM occurs (1B results are sufficient for scaling law demonstration)
  - See `docs/gitignore/memo_catalyst_oom_and_scaling.md` for detailed OOM analysis

**Current Status (as of 2026-01-30):**
- ✅ 70M, 160M, 410M, 1B: Successfully run on 24GB GPU
- ⚠️ 1.4B: OOM during attack loop (catalyst generation succeeds)
- ❓ 2.8B: Not tested yet, likely to OOM

## Command-Line Options

- `--model {70m,160m,410m,1b,1.4b,2.8b}`: Model size
- `--quantization {none,fp16,4bit,8bit}`: Quantization mode (use `fp16` for Priority 1)
- `--num-runs N`: Number of repetitions (use `5` for Priority 1)
- `--seed N`: Base seed (each run will use `N + run_number`)
- `--force-fft`: Enable Full Fine-Tuning mode (required for Priority 1)
- `--skip-control`: Skip control group (only run treatment group)
- `--verbosity {quiet,normal,verbose}`: Output verbosity level
- `--output-dir PATH`: Output directory for results

## Data Analysis

After running experiments, use `analyze_priority1.py` to generate scaling law visualizations and summary statistics.

### Basic Usage

Generate scaling law analysis without outlier filtering:

```bash
python experiments/01_pythia_160m/analyze_priority1.py \
    --results-dir ./experiments/results/ \
    --models 70m 160m 410m 1b \
    --output-dir ./experiments/results/scaling_law_analysis/
```

### With Outlier Filtering

Filter outliers using **IQR (Interquartile Range)** method based on perplexity increase:

```bash
python experiments/01_pythia_160m/analyze_priority1.py \
    --results-dir ./experiments/results/ \
    --models 70m 160m 410m 1b \
    --output-dir ./experiments/results/scaling_law_analysis/ \
    --filter-method iqr \
    --filter-metric perplexity_increase \
    --filter-iqr-multiplier 1.5
```

Filter outliers using **Z-score** method based on rank reduction:

```bash
python experiments/01_pythia_160m/analyze_priority1.py \
    --results-dir ./experiments/results/ \
    --models 70m 160m 410m 1b \
    --output-dir ./experiments/results/scaling_law_analysis/ \
    --filter-method zscore \
    --filter-metric rank_reduction \
    --filter-zscore-threshold 3.0
```

### Analysis Script Options

- `--results-dir PATH`: Path to results directory (default: `./experiments/results/`)
- `--models MODEL1 MODEL2 ...`: List of models to analyze (default: `70m 160m 410m 1b`)
- `--output-dir PATH`: Output directory for plots and reports (default: `./experiments/results/scaling_law_analysis/`)
- `--use-absolute`: Use absolute values for structural damage (default: True)
- `--no-absolute`: Use signed values for structural damage
- `--filter-method {iqr,zscore}`: Outlier filtering method
- `--filter-metric {rank_reduction,perplexity_increase,perplexity_post}`: Metric to filter on (default: `rank_reduction`)
- `--filter-iqr-multiplier FLOAT`: IQR multiplier for outlier detection (default: 1.5)
- `--filter-zscore-threshold FLOAT`: Z-score threshold for outlier detection (default: 3.0)

### Output Files

The analysis script generates:

- `scaling_law_summary.csv`: Summary statistics table (CSV format)
- `scaling_law_summary.json`: Summary statistics with filtering metadata (JSON format)
- `scaling_law_curve.png`: Scaling law visualization plot

**Filtering Information:**

When filtering is applied, the JSON output includes filtering metadata:
- Number of filtered outliers per model
- Filtering method and metric used
- Indices of removed runs
- Original vs. filtered statistics

## Results

Results are saved to `experiments/results/` with the following structure:
- `pythia_<SIZE>_pythia-<SIZE>_run<NN>_fp16/`: Individual run results
- `aggregate_summary_*.csv`: Aggregated statistics across runs
- `scaling_law_analysis/`: Analysis outputs (summary tables, plots)

Results will be documented here as experiments progress.
