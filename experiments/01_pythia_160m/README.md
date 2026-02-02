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

## Script Architecture

This experimental suite uses a two-layer architecture:

- **`experiment_orchestrator.py`**: **Entry point** for Priority experiments. Automatically configures and runs all experiments for a given priority level.
- **`run_experiment.py`**: **Infrastructure** script that executes individual experiments. Called internally by the orchestrator.

**Recommended Usage**: Use `experiment_orchestrator.py` for Priority experiments (1-4). Use `run_experiment.py` directly only for single runs or advanced custom configurations.

## Experimental Strategy: The Four Priorities

This experimental suite is designed to build a **defensive evidence chain** that addresses critical reviewer questions. The experiments are organized into four priorities, each targeting a specific scientific question.

### Core Strategy: Defensive Experimental Design

We address three critical reviewer questions:
1. **"Is it just coincidence?"** → Priority 1: Scaling Law
2. **"Would any garbage data cause this?"** → Priority 2: Placebo Test
3. **"Is it the optimizer's fault?"** → Priority 3: Mechanism Test
4. **"How thick should the shield be?"** → Priority 4: Shield Matrix

---

### Priority 1: Scaling Law (The "Royal Flush")

**What**: Generate the core scaling law curve showing **Model Size (X-axis) vs. Structural Damage (Y-axis)**.

**Why**: Reviewers will question: *"You only tested three models (70M, 160M, 410M). The sample size is too small; the trend is unreliable."*

**How**: 
- **Models**: Pythia-70M, 160M, 410M, 1B (and optionally 1.4B, 2.8B)
- **Precision**: FP16 (unified, no quantization) to control variables
- **Mode**: Full Fine-Tuning (FFT) to enable full parameter updates
- **Repetitions**: 5 runs per model (sufficient for mean and std dev)
- **Total**: 4-6 models × 5 runs = 20-30 experiments

**Expected Result**: An exponential curve showing that larger models are more vulnerable to structural attacks (inverse scaling law).

**Key Finding**: The 1B model shows catastrophic structural collapse (~26.5% rank reduction), confirming the exponential vulnerability scaling.

---

### Priority 2: Placebo Test (Treatment Specificity)

**What**: Prove that "Eigen-Prion" is a special attack vector, not just random noise.

**Why**: Reviewers will question: *"Large models are naturally forgetful. If you feed 100 random garbage tokens (Gaussian Noise) or random text, wouldn't they also collapse?"*

**How**:
- **Model**: Pythia-410M (FP16) as a representative medium-sized model
- **Control A (Random Noise)**: Generate Gaussian noise vectors of equivalent dimensions
- **Control B (Random Text)**: Random text samples from the Pile dataset
- **Treatment (Eigen-Prion)**: Our structured attack data
- **Repetitions**: 3 runs per treatment (sufficient if controls show near-zero effect)

**Expected Result**: Control groups show minimal rank change (~0%), while the treatment group shows significant structural damage. This proves **structural specificity** of the attack.

---

### Priority 3: Mechanism Test (Optimizer Verification)

**What**: Validate the "Adam metabolic amplification" theory.

**Why**: Reviewers will question: *"You claim Adam causes this. If you switch to SGD, shouldn't it be fine? If true, your theory holds."*

**How**:
- **Model**: Pythia-410M (FP16)
- **Optimizer A**: AdamW (standard configuration with adaptive learning rates)
- **Optimizer B**: SGD (no momentum, no adaptive learning rate)
- **Repetitions**: 3 runs per optimizer

**Expected Result**: SGD group shows **attack failure** (model immune), while AdamW group shows **attack success**. This validates the theoretical mechanism (Lemma 1).

---

### Priority 4: Shield Matrix (Quantization Boundary Exploration)

**What**: Extend the "quantization immunity" finding into a complete conclusion about defense mechanisms.

**Why**: Reviewers will question: *"You claim quantization is a shield. What about 8-bit? How thick should the shield be to be effective?"*

**How**:
- **Model**: Pythia-1B (or 410M - a model that collapses under FP16)
- **Precision A**: FP16 (baseline, collapses)
- **Precision B**: 8-bit (BitsAndBytes Int8)
- **Precision C**: 4-bit (NF4)
- **Mode**: QLoRA (Quantized LoRA) for quantized models (FFT incompatible with discrete weights)
- **Repetitions**: 3 runs per precision

**Expected Result**: FP16 shows collapse > 8-bit shows minor damage/immunity > 4-bit shows complete immunity. This demonstrates the **shield thickness effect**.

**Technical Note**: Quantized models use QLoRA instead of FFT because:
- **Physics**: Gradients require continuous weights; quantized weights are discrete (Int8/Int4)
- **Solution**: QLoRA allows metabolic updates through low-rank adapters while the backbone remains frozen
- **Interpretation**: Quantization acts as a **structural firewall**, forcing updates into a constrained low-rank space

---

### Summary: Experimental Workflow

The priorities are designed to be executed sequentially:

1. **[Highest Priority] Priority 1**: Establish the scaling law curve (foundational evidence)
2. **[Medium Priority] Priority 2 & 3**: Validate specificity and mechanism (control experiments)
3. **[Lower Priority] Priority 4**: Explore defense mechanisms (extension finding)

**Total Workload**: ~50-60 experimental runs, manageable on a single RTX 4090 over 2-3 days.

## Running experiments

### Quick Start: Using the Orchestrator (Recommended)

The easiest way to run Priority experiments is using `experiment_orchestrator.py`:

**Priority 1: Scaling Law**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 1 --num-runs 10
```

**Priority 2: Placebo Test**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 2 --num-runs 3
```

**Priority 3: Mechanism Test**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 3 --num-runs 3
```

**Priority 4: Shield Matrix**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 4 --model 1b --num-runs 3
```

The orchestrator automatically:
- Configures the correct models, precisions, and optimizers for each priority
- Sets up proper output directories (`priority<N>_<name>/raw_data/`)
- Handles FFT/QLoRA mode selection based on quantization
- Manages random seeds across runs

### Priority 1: Scaling Law Experiments

**Purpose**: Generate the core scaling law curve (Model Size vs. Structural Damage) to demonstrate inverse scaling vulnerability.

**Reviewer Question Addressed**: *"Is the trend real, or just coincidence with too few data points?"*

**Using the Orchestrator (Recommended):**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 1 --num-runs 10
```

This automatically runs all models (70m, 160m, 410m, 1b) with FP16 precision and FFT mode.

**Manual Execution (Advanced):**

If you need custom configuration, you can use `run_experiment.py` directly:

```bash
python experiments/01_pythia_160m/run_experiment.py \
    --model 70m \
    --quantization fp16 \
    --force-fft \
    --num-runs 10 \
    --seed 42 \
    --output-dir ./experiments/results/priority1_scaling_law/raw_data/ \
    --control-type none \
    --verbosity normal
```

**Models:** 70m, 160m, 410m, 1b (1.4b, 2.8b optional but may OOM)  
**Total:** 4 models × 10 runs = 40 experiments

**Note on seed behavior:** When using `--num-runs N --seed BASE_SEED`, each run will automatically use a different seed:
- Run 1: `BASE_SEED + 1`
- Run 2: `BASE_SEED + 2`
- Run 3: `BASE_SEED + 3`
- ... and so on

This ensures each repetition uses a different random seed for proper statistical analysis.

### Priority 2: Placebo Test Experiments

**Purpose**: Compare treatment specificity (Eigen-Prion vs controls) to prove structural attack specificity.

**Reviewer Question Addressed**: *"Would any garbage data cause structural collapse, or is Eigen-Prion special?"*

**Using the Orchestrator (Recommended):**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 2 --num-runs 3
```

This automatically runs all treatments (eigen_prion, gaussian_noise, random_text) on the 410m model.

### Priority 3: Mechanism Test Experiments

**Purpose**: Compare optimizer effects (AdamW vs SGD) to validate the "Adam metabolic amplification" theory.

**Reviewer Question Addressed**: *"Is it the optimizer's fault? If SGD prevents collapse, your theory holds."*

**Using the Orchestrator (Recommended):**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 3 --num-runs 3
```

This automatically runs both optimizers (adamw, sgd) on the 410m model.

### Priority 4: Shield Matrix Experiments

**Purpose**: Compare quantization defense effects (FP16 vs 8-bit vs 4-bit) to explore the "quantization shield" mechanism.

**Reviewer Question Addressed**: *"You claim quantization is a shield. How thick should it be? What about 8-bit?"*

**Using the Orchestrator (Recommended):**
```bash
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 4 --model 1b --num-runs 3
```

This automatically runs all precisions (fp16, 8bit, 4bit) on the specified model, using FFT for FP16 and QLoRA for quantized models.

**Technical Note**: Quantized models (8-bit, 4-bit) use **QLoRA** instead of FFT because:
- Discrete weights (Int8/Int4) cannot compute gradients directly
- QLoRA allows metabolic updates through low-rank adapters while the quantized backbone remains frozen
- This demonstrates that quantization acts as a **structural firewall**, constraining updates to a low-rank space

### Single Run (Advanced Usage)

For single experimental runs or custom configurations, use `run_experiment.py` directly:

```bash
python experiments/01_pythia_160m/run_experiment.py \
    --model 160m \
    --quantization fp16 \
    --force-fft \
    --control-type none \
    --output-dir ./experiments/results/
```

See `python experiments/01_pythia_160m/run_experiment.py --help` for all available options.

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

### Experiment Orchestrator (`experiment_orchestrator.py`)

**Primary entry point for Priority experiments:**

- `--priority {1,2,3,4}`: Priority level to run
  - `1`: Scaling Law (runs all models: 70m, 160m, 410m, 1b)
  - `2`: Placebo Test (runs all treatments: eigen_prion, gaussian_noise, random_text)
  - `3`: Mechanism Test (runs both optimizers: adamw, sgd)
  - `4`: Shield Matrix (runs all precisions: fp16, 8bit, 4bit)
- `--num-runs N`: Number of runs per experiment (default: 10 for Priority 1, 3 for Priorities 2-4)
- `--seed N`: Base random seed (default: 42)
- `--model {70m,160m,410m,1b,1.4b,2.8b}`: Model to use (only for Priority 4, default: 1b)
- `--output-dir PATH`: Base output directory (default: `./experiments/results/`)
- `--verbosity {quiet,normal,verbose}`: Output verbosity level (default: normal)

**Examples:**
```bash
# Priority 1: Scaling Law
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 1 --num-runs 10

# Priority 2: Placebo Test
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 2

# Priority 3: Mechanism Test
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 3

# Priority 4: Shield Matrix (with custom model)
python experiments/01_pythia_160m/experiment_orchestrator.py --priority 4 --model 1b
```

### Run Experiment (`run_experiment.py`)

**Infrastructure script for individual experiments (called by orchestrator):**

- `--model {70m,160m,410m,1b,1.4b,2.8b}`: Model size
- `--quantization {none,fp16,4bit,8bit}`: Quantization mode
- `--num-runs N`: Number of repetitions
- `--seed N`: Base seed (each run uses `N + run_number`)
- `--force-fft`: Enable Full Fine-Tuning mode (required for Priority 1)
- `--optimizer {adamw,sgd}`: Optimizer choice (for Priority 3)
- `--control-type {none,random_tokens,gaussian_noise,random_text,eigen_prion}`: Treatment type
- `--verbosity {quiet,normal,verbose}`: Output verbosity level
- `--output-dir PATH`: Output directory for results

**Note**: For Priority experiments, use `experiment_orchestrator.py` instead. Use `run_experiment.py` directly only for single runs or advanced custom configurations.

## Data Analysis

After running experiments, use `analyze_experiments.py` (or the backward-compatible `analyze_priority1.py`) to analyze data across all priority types.

### Priority 1: Scaling Law Analysis

**Purpose**: Generate scaling law curve (Model Size vs Structural Damage)

**Basic usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 1 \
    --results-dir ./experiments/results/priority1_scaling_law/raw_data/ \
    --models 70m 160m 410m 1b \
    --output-dir ./experiments/results/priority1_scaling_law/
```

**With outlier filtering**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 1 \
    --results-dir ./experiments/results/priority1_scaling_law/raw_data/ \
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

**Purpose**: Compare treatment specificity (Eigen-Prion vs controls) to prove structural attack specificity.

**Reviewer Question Addressed**: *"Would any garbage data cause structural collapse, or is Eigen-Prion special?"*

**Usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 2 \
    --results-dir ./experiments/results/priority2_placebo/raw_data/ \
    --model 410m \
    --treatments eigen_prion gaussian_noise random_text \
    --output-dir ./experiments/results/priority2_placebo/
```

**Output files**:
- `placebo_summary.csv`: Treatment comparison statistics
- `placebo_summary.json`: Statistical test results (ANOVA, pairwise t-tests)
- `placebo_comparison.png`: Bar chart comparing treatments with significance markers

### Priority 3: Mechanism Test Analysis

**Purpose**: Compare optimizer effects (AdamW vs SGD) to validate the "Adam metabolic amplification" theory.

**Reviewer Question Addressed**: *"Is it the optimizer's fault? If SGD prevents collapse, your theory holds."*

**Usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 3 \
    --results-dir ./experiments/results/priority3_mechanism/raw_data/ \
    --model 410m \
    --optimizers adamw sgd \
    --output-dir ./experiments/results/priority3_mechanism/
```

**Output files**:
- `mechanism_summary.csv`: Optimizer comparison statistics
- `mechanism_summary.json`: Statistical test results (t-test)
- `mechanism_comparison.png`: Side-by-side bar chart with p-value annotation

### Priority 4: Shield Matrix Analysis

**Purpose**: Compare quantization defense effects (FP16 vs 8-bit vs 4-bit) to explore the "quantization shield" mechanism.

**Reviewer Question Addressed**: *"You claim quantization is a shield. How thick should it be? What about 8-bit?"*

**Technical Note**: Quantized models (8-bit, 4-bit) use **QLoRA** instead of FFT because:
- Discrete weights (Int8/Int4) cannot compute gradients directly
- QLoRA allows metabolic updates through low-rank adapters while the quantized backbone remains frozen
- This demonstrates that quantization acts as a **structural firewall**, constraining updates to a low-rank space

**Usage**:
```bash
python experiments/01_pythia_160m/analyze_experiments.py 4 \
    --results-dir ./experiments/results/priority4_shield/raw_data/ \
    --model 1b \
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
- `--results-dir PATH`: Path to raw experiment data directory (typically `./experiments/results/priority<N>_<name>/raw_data/`)
- `--output-dir PATH`: Output directory for plots and reports (typically `./experiments/results/priority<N>_<name>/`)

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

**Directory organization**:
- `priority1_scaling_law/`: Scaling law analysis
  - `raw_data/`: Raw experiment data (individual runs, aggregate CSVs)
  - `scaling_law_summary.csv`, `scaling_law_summary.json`, `scaling_law_curve.png`: Analysis outputs
- `priority2_placebo/`: Placebo test analysis
  - `raw_data/`: Raw experiment data
  - `placebo_summary.csv`, `placebo_summary.json`, `placebo_comparison.png`: Analysis outputs
- `priority3_mechanism/`: Mechanism test analysis
  - `raw_data/`: Raw experiment data
  - `mechanism_summary.csv`, `mechanism_summary.json`, `mechanism_comparison.png`: Analysis outputs
- `priority4_shield/`: Shield matrix analysis
  - `raw_data/`: Raw experiment data
  - `shield_summary.csv`, `shield_summary.json`, `shield_matrix.png`: Analysis outputs

**Raw data structure** (within each `raw_data/` directory):
- `pythia_<SIZE>_pythia-<SIZE>_run<NN>_<PRECISION>/`: Individual run results
- `aggregate_summary_*.csv`: Aggregated statistics across runs

Results will be documented here as experiments progress.
