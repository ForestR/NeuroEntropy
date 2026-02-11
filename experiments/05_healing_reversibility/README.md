# Healing/Reversibility Experiment (Priority 5)

Tests whether spectral collapse induced by HAEP (Hessian-Aware Eigen-Perturbation) attacks can be reversed through standard supervised fine-tuning (SFT) on clean data.

## Overview

This experiment addresses Reviewer #2's critique: *"If the model heals quickly, it's not fibrosis; it's just noise injection."* A true pathological state must be a stable local minimum that is hard to escape.

**Protocol:**
1. Load a collapsed model (from checkpoint or run fresh HAEP attack)
2. Measure collapsed state metrics (effective rank, perplexity, spectral gap)
3. Run SFT on clean Pile data for ~500 steps
4. Track recovery trajectory every 10 steps
5. Generate visualizations and compute recovery statistics

## Usage

### Option 1: Use existing collapsed checkpoint (single)

If you have a post-attack checkpoint from a previous run (e.g., Priority 1 scaling law with `--save-checkpoints`):

```bash
python experiments/05_healing_reversibility/run_healing_experiment.py \
    --model 1b \
    --collapsed-checkpoint experiments/results/priority1_scaling_law/raw_data/pythia_1b_pythia-1b_run01_fp16/checkpoints/checkpoint_post_treatment.pt \
    --healing-steps 500 \
    --healing-lr 1e-4
```

### Option 2: Multi-checkpoint mode (scan directory)

Scan a directory for all `checkpoint_post_treatment.pt` files and run healing on each. Useful when only some runs saved checkpoints:

```bash
python experiments/05_healing_reversibility/run_healing_experiment.py \
    --model 1b \
    --checkpoint-dir experiments/results/priority1_scaling_law/raw_data \
    --runs-per-checkpoint 2 \
    --healing-steps 500 \
    --healing-lr 1e-4
```

### Option 3: Run fresh attack then heal

If no checkpoint is available, the script will run a fresh HAEP attack first, then attempt healing:

```bash
python experiments/05_healing_reversibility/run_healing_experiment.py \
    --model 1b \
    --healing-steps 500 \
    --healing-optimizer sgd  # Test if SGD heals better than AdamW
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | 1b | Model size (70m, 160m, 410m, 1b, 1.4b, 2.8b) |
| `--collapsed-checkpoint` | None | Path to single collapsed checkpoint |
| `--checkpoint-dir` | None | Directory to scan for `checkpoint_post_treatment.pt` |
| `--runs-per-checkpoint` | 1 | Runs per checkpoint when using `--checkpoint-dir` (1–3) |
| `--healing-steps` | 500 | Number of SFT steps for healing attempt |
| `--healing-lr` | 1e-4 | Learning rate for healing SFT |
| `--healing-optimizer` | adamw | Optimizer (adamw or sgd) |
| `--num-runs` | 1 | Number of replicates when using single checkpoint |
| `--gradient-checkpointing` | off | Enable to save memory (may increase NaN on collapsed models) |
| `--output-dir` | ./experiments/results | Base output directory |
| `--verbosity` | normal | Output level (quiet, normal, verbose) |

## Expected Outputs

Results are saved to `experiments/results/priority5_healing/raw_data/<experiment_name>/`:

- `metrics.jsonl` - Step-by-step healing metrics
- `config.json` - Experiment configuration
- `healing_summary.json` - Final recovery statistics
- `recovery_trajectory.png` - Effective rank over healing steps
- `perplexity_recovery.png` - Perplexity over healing steps (log scale)
- `spectral_decay_three_state.png` - Three-state spectral comparison (healthy vs collapsed vs healed)

## Interpretation

### Scenario A: Irreversible (Model stays dead)

- Final rank remains near collapsed state
- Perplexity stays high
- Recovery percentage < 20%
- **Conclusion**: "Irreversible Pathology" - supports fibrosis analogy

### Scenario B: Reversible (Model heals)

- Rank recovers toward original baseline
- Perplexity decreases
- Recovery percentage > 70%
- **Conclusion**: "Acute Structural Failure" - still valuable (DoS threat)

### Recovery Metrics

- **Rank recovery %**: `(final_rank - collapsed_rank) / (original_rank - collapsed_rank) * 100`
- **PPL improvement %**: `(collapsed_ppl - final_ppl) / collapsed_ppl * 100`

## Requirements

- Same dependencies as main experiment (`transformers`, `torch`, `datasets` for Pile)
- CUDA recommended for 1B model
- ~2–4 hours for full 500-step healing on 1B (depending on hardware)

## Integration with Paper

Results update `paper/sections/05_discussion.tex` section 5.5 "Addressing Irreversibility":

- Replace "We have not run this experiment" with empirical findings
- Add recovery trajectory figure to Results section
- Support either "Irreversible Pathology" or "Acute Structural Failure" narrative
