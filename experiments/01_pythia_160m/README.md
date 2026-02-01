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

## Experimental Configuration & Optimization Strategy

### Parameter Configuration

**Unified Configuration** (all models):
- **Fine-tuning Mode**: Full Fine-Tuning (FFT) with `--force-fft`
- **Precision**: FP16 (`--quantization fp16`) - no quantization to avoid confounding factors
- **Gradient Checkpointing**: Enabled automatically with FFT mode
- **Learning Rate**: 1e-4 (70M/160M/410M), 2e-4 (1B+) - higher LR for larger models to compensate for relative perturbation reduction
- **Catalyst Length**: 64 tokens (reduced from 128 to save activation memory)
- **Attack Steps**: 100 iterations
- **Random Seeds**: Base seed 42, each run uses `42 + run_number` (Run 1→43, Run 2→44, ...)

**Key Design Principles**:
- ✅ **Configuration Consistency**: All models use identical settings (except learning rate for 1B+) to ensure comparability
- ✅ **FFT Mode**: Enables full parameter updates, ensuring attack effectiveness
- ✅ **FP16 Precision**: Avoids quantization artifacts while maintaining memory efficiency

### Optimization Strategy

To enable large models (up to 1B) to run on 24GB GPUs, we apply a **layered optimization strategy**:

**Base Optimizations** (all models):
- **CPU Offloading**: Null directions stored on CPU (~14GB saved for 1B)
- **FP16 HVP**: Hessian-vector products computed in FP16 (~2GB saved)
- **Adafactor Optimizer**: Zero optimizer state overhead (~11.2GB saved for 1B)
- **Gradient Checkpointing**: Reduces activation memory (~3-4GB saved)
- **Catalyst Length=64**: Reduced from 128 (~4-5GB saved)

**Large Model Optimizations** (>1B):
- **Reduced Null Directions**: 10 → 3 directions (~14GB saved for 1B)
- **Learning Rate Boost**: 1e-4 → 2e-4 to compensate for relative perturbation reduction

**Total Memory Savings** (1B model): ~48-52GB, enabling successful execution on 24GB GPU.

### Expected Results

**Scaling Law Observation**:
The experiments demonstrate an **inverse scaling law** - larger models are more vulnerable to structural attacks:

- **70M**: ~3% rank reduction (noise level)
- **160M**: ~4% rank reduction (minor damage)
- **410M**: ~5.8% rank reduction (moderate damage)
- **1B**: **~26.5% rank reduction** (catastrophic structural collapse)

**Key Findings**:
- ✅ **Exponential vulnerability scaling**: The damage curve follows a J-curve, not linear
- ✅ **High variance in large models**: 1B model shows high standard deviation (13.27%), indicating "rugged null space" - sometimes finds "cliffs" (46% drop), sometimes "slopes" (4% drop)
- ✅ **Perplexity explosion**: Post-attack perplexity increases dramatically (e.g., 1B: ~29,000% increase), indicating total functional loss

**Visualization**:
The analysis script generates a scaling law plot showing:
- **X-axis**: Model size (log scale)
- **Y-axis**: Mean rank reduction percentage
- **Error bars**: Standard deviation
- **Trend**: Upward exponential curve confirming inverse scaling law

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

After running experiments, use `analyze_experiments.py` (or the backward-compatible `analyze_priority1.py`) to analyze data across all priority types.

### Priority 1: Scaling Law Analysis

**Purpose**: Generate scaling law curve (Model Size vs Structural Damage)

**Basic usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 1 \
    --results-dir ./experiments/results/ \
    --models 70m 160m 410m 1b \
    --output-dir ./experiments/results/priority1_scaling_law/
```

**With outlier filtering**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 1 \
    --results-dir ./experiments/results/ \
    --models 70m 160m 410m 1b \
    --output-dir ./experiments/results/priority1_scaling_law/ \
    --filter-method iqr \
    --filter-metric perplexity_increase \
    --filter-iqr-multiplier 1.5
```

**Output files**:
- `scaling_law_summary.csv`: Summary statistics table
- `scaling_law_summary.json`: Summary statistics with filtering metadata
- `scaling_law_curve.png`: Scaling law visualization plot

### Priority 2: Placebo Test Analysis

**Purpose**: Compare treatment specificity (Eigen-Prion vs controls)

**Usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 2 \
    --results-dir ./experiments/results/ \
    --model 410m \
    --treatments eigen_prion gaussian_noise random_text \
    --output-dir ./experiments/results/priority2_placebo/
```

**Output files**:
- `placebo_summary.csv`: Treatment comparison statistics
- `placebo_summary.json`: Statistical test results (ANOVA, pairwise t-tests)
- `placebo_comparison.png`: Bar chart comparing treatments with significance markers

### Priority 3: Mechanism Test Analysis

**Purpose**: Compare optimizer effects (AdamW vs SGD)

**Usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 3 \
    --results-dir ./experiments/results/ \
    --model 410m \
    --optimizers adamw sgd \
    --output-dir ./experiments/results/priority3_mechanism/
```

**Output files**:
- `mechanism_summary.csv`: Optimizer comparison statistics
- `mechanism_summary.json`: Statistical test results (t-test)
- `mechanism_comparison.png`: Side-by-side bar chart with p-value annotation

### Priority 4: Shield Matrix Analysis

**Purpose**: Compare quantization defense effects (FP16 vs 8-bit vs 4-bit)

**Usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 4 \
    --results-dir ./experiments/results/ \
    --model 1.4b \
    --precisions fp16 8bit 4bit \
    --output-dir ./experiments/results/priority4_shield/
```

**Output files**:
- `shield_summary.csv`: Precision comparison statistics
- `shield_summary.json`: Statistical test results (ANOVA, pairwise t-tests)
- `shield_matrix.png`: Bar chart showing quantization shield effectiveness

### Backward Compatibility

The old `analyze_priority1.py` script still works and forwards to the new `analyze_experiments.py`:

```bash
# Old command still works
python experiments/01_pythia_160m/analyze_priority1.py \
    --results-dir ./experiments/results/ \
    --models 70m 160m 410m 1b \
    --output-dir ./experiments/results/scaling_law_analysis/
```

### Analysis Script Options

**Common options** (all priorities):
- `--results-dir PATH`: Path to results directory (default: `./experiments/results/`)
- `--output-dir PATH`: Output directory for plots and reports (required)

**Priority 1 specific**:
- `--models MODEL1 MODEL2 ...`: List of models to analyze (required)
- `--use-absolute`: Use absolute values for structural damage (default: True)
- `--no-absolute`: Use signed values for structural damage
- `--filter-method {iqr,zscore}`: Outlier filtering method
- `--filter-metric {rank_reduction,perplexity_increase,perplexity_post}`: Metric to filter on
- `--filter-iqr-multiplier FLOAT`: IQR multiplier for outlier detection (default: 1.5)
- `--filter-zscore-threshold FLOAT`: Z-score threshold for outlier detection (default: 3.0)

**Priority 2 specific**:
- `--model MODEL`: Model name to analyze (required, e.g., "410m")
- `--treatments TREATMENT1 TREATMENT2 ...`: Treatment types to compare (default: `eigen_prion gaussian_noise random_text`)

**Priority 3 specific**:
- `--model MODEL`: Model name to analyze (required)
- `--optimizers OPTIMIZER1 OPTIMIZER2 ...`: Optimizers to compare (default: `adamw sgd`)

**Priority 4 specific**:
- `--model MODEL`: Model name to analyze (required)
- `--precisions PRECISION1 PRECISION2 ...`: Precision levels to compare (default: `fp16 8bit 4bit`)

## Results

Results are saved to `experiments/results/` with the following structure:
- `pythia_<SIZE>_pythia-<SIZE>_run<NN>_fp16/`: Individual run results
- `aggregate_summary_*.csv`: Aggregated statistics across runs
- `scaling_law_analysis/`: Analysis outputs (summary tables, plots)

Results will be documented here as experiments progress.
