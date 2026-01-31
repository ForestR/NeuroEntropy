#!/usr/bin/env python3
"""
Single Model Experiment Runner

Runs metabolic attack experiment on a single Pythia model.
Implements the protocol from Phase II of the roadmap:
1. Baseline measurement
2. Control group (random noise)
3. Treatment group (Eigen-Prion catalyst)
4. Post-attack measurement
"""

import re
import sys
import os
import torch
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.catalyst import HessianAwareCatalyst
from src.attack_loop import MetabolicAttackLoop
from src.diagnosis import diagnose_model_health, ActivationHook
from src.config import (
    ExperimentConfig,
    get_pythia_160m_config,
    get_pythia_70m_config,
    get_pythia_410m_config,
    get_pythia_1b_config,
    get_pythia_1_4b_config,
    get_pythia_2_8b_config,
)
from src.utils.logger import ExperimentLogger
from src.utils.visualizer import plot_rank_collapse, plot_rank_reduction, plot_spectral_decay
from experiments.utils.evaluate import compute_perplexity, create_simple_test_set


def vprint(message: str, level: str = 'normal', verbosity: str = 'normal'):
    """Verbosity-aware print function."""
    levels = {'quiet': 0, 'normal': 1, 'verbose': 2}
    if levels.get(verbosity, 1) >= levels.get(level, 1):
        print(message)


def load_model(config: ExperimentConfig, device: str, verbosity: str = 'normal'):
    """Load model and tokenizer."""
    vprint(f"Loading model: {config.model.hf_model_id}", 'normal', verbosity)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional quantization
    if config.model.use_quantization:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True if config.model.quantization_bits == 4 else False,
            load_in_8bit=True if config.model.quantization_bits == 8 else False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model.hf_model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.hf_model_id
        ).to(device)
    
    # Enable gradient checkpointing for FFT mode (required for 1B+ models)
    if getattr(config.model, 'use_gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            vprint("Gradient checkpointing enabled", 'normal', verbosity)
        else:
            vprint("Warning: Model does not support gradient checkpointing", 'quiet', verbosity)
    
    # Enable all parameters for training in FFT mode
    if getattr(config.model, 'use_fft', False):
        for param in model.parameters():
            param.requires_grad = True
        vprint("Full Fine-Tuning mode: All parameters trainable", 'normal', verbosity)
    
    model.eval()
    return model, tokenizer


def measure_baseline(
    model,
    tokenizer,
    device: str,
    num_samples: int = 10,
    verbosity: str = 'normal'
) -> dict:
    """Measure baseline metrics."""
    vprint("Measuring baseline metrics...", 'normal', verbosity)
    
    # Create test inputs
    test_texts = create_simple_test_set(num_samples)
    
    # Measure perplexity
    perplexity = compute_perplexity(model, tokenizer, test_texts, device)
    
    # Measure effective rank using activation hook
    hook = ActivationHook()
    try:
        hook.register(model)
        
        # Collect activations from a few forward passes
        model.eval()
        with torch.no_grad():
            for i in range(min(5, len(test_texts))):
                inputs = tokenizer(
                    test_texts[i],
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding='max_length'  # Ensure consistent length
                ).to(device)
                _ = model(**inputs)
        
        activations = hook.get_activations()
        hook.clear()
        
        if activations is not None:
            from src.diagnosis import compute_effective_rank, compute_spectral_gap, extract_singular_values
            effective_rank = compute_effective_rank(activations)
            spectral_gap = compute_spectral_gap(activations)
            # Extract singular values for spectral decay plot
            singular_values = extract_singular_values(activations, normalize=True)
        else:
            effective_rank = 0.0
            spectral_gap = 0.0
            singular_values = np.array([])
    except Exception as e:
        vprint(f"Warning: Could not measure effective rank: {e}", 'quiet', verbosity)
        effective_rank = 0.0
        spectral_gap = 0.0
        singular_values = np.array([])
    
    return {
        'perplexity': perplexity,
        'effective_rank': effective_rank,
        'spectral_gap': spectral_gap,
        'singular_values': singular_values
    }


def run_control_group(
    model,
    tokenizer,
    device: str,
    num_steps: int = 100,
    learning_rate: float = 1e-4,
    verbosity: str = 'normal'
) -> list:
    """Run control group: inject random noise."""
    vprint(f"Running control group: {num_steps} steps of random noise...", 'normal', verbosity)
    
    model.train()
    history = []
    
    vocab_size = len(tokenizer)
    
    for step in range(num_steps):
        # Generate random tokens
        random_tokens = torch.randint(
            0, vocab_size,
            (1, 128),
            device=device
        )
        
        # Forward and backward pass
        labels = random_tokens[:, 1:].contiguous()
        input_ids = random_tokens[:, :-1].contiguous()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Update parameters
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
        
        # Measure rank periodically
        if step % 10 == 0:
            model.eval()
            hook = ActivationHook()
            try:
                hook.register(model)
                with torch.no_grad():
                    _ = model(input_ids=random_tokens)
                activations = hook.get_activations()
                hook.clear()
                
                if activations is not None:
                    from src.diagnosis import compute_effective_rank
                    rank = compute_effective_rank(activations)
                else:
                    rank = 0.0
            except Exception:
                rank = 0.0
            
            history.append({
                'iteration': step,
                'effective_rank': rank,
                'loss': loss.item()
            })
            model.train()
    
    model.eval()
    return history


def run_treatment_group(
    model,
    tokenizer,
    device: str,
    config: ExperimentConfig,
    verbosity: str = 'normal'
) -> dict:
    """Run treatment group: inject Eigen-Prion catalyst."""
    vprint(
        f"Running treatment group: {config.attack.num_steps} steps of Eigen-Prion...",
        'normal',
        verbosity,
    )
    
    # Initialize catalyst generator
    catalyst_gen = HessianAwareCatalyst(
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k_eigenvalues=config.hvp.num_null_directions
    )
    
    # Initialize attack loop
    attack_loop = MetabolicAttackLoop(
        model=model,
        catalyst_generator=catalyst_gen,
        device=device
    )
    
    # Run attack
    results = attack_loop.run_attack_cycle(
        num_iterations=config.attack.num_steps,
        target_rank_reduction=config.attack.target_rank_reduction,
        learning_rate=config.attack.learning_rate,
        catalyst_length=config.attack.catalyst_length
    )
    
    return {
        'results': results,
        'history': attack_loop.get_history()
    }


def run_single_experiment(
    config: ExperimentConfig,
    args: argparse.Namespace,
    run_number: int,
    verbosity: str,
) -> dict:
    """
    Run a single experiment iteration.

    Returns:
        dict: Results containing baseline, post_attack metrics, and rank_reduction
    """
    # Set seed for reproducibility
    torch.manual_seed(config.seed + run_number)
    np.random.seed(config.seed + run_number)

    # Create run-specific experiment name
    experiment_name = f"pythia_{args.model}_{config.model.hf_model_id.split('/')[-1]}"
    if args.num_runs > 1:
        experiment_name += f"_run{run_number:02d}"

    # Add quantization suffix
    if config.model.use_quantization:
        experiment_name += f"_{config.model.quantization_bits}bit"
    else:
        experiment_name += "_fp16"

    # Initialize logger
    logger = ExperimentLogger(
        output_dir=config.output_dir,
        experiment_name=experiment_name,
    )

    logger.log_config({
        'model': config.model.__dict__,
        'attack': config.attack.__dict__,
        'hvp': config.hvp.__dict__,
    })

    # Load model
    model, tokenizer = load_model(config, args.device, verbosity=verbosity)

    # Keep baseline state in memory when not saving checkpoints (for reset before treatment)
    baseline_state_dict = None

    # Resume from checkpoint if specified, otherwise save baseline checkpoint
    if args.resume_from_checkpoint:
        if not args.save_checkpoints:
            vprint("Note: --save-checkpoints is False; resuming from specified path only.", 'quiet', verbosity)
        logger.load_checkpoint(
            checkpoint_path=args.resume_from_checkpoint,
            model=model
        )
        vprint(f"Resumed from checkpoint: {args.resume_from_checkpoint}", 'quiet', verbosity)
    else:
        # Save initial model state (baseline checkpoint) only if --save-checkpoints
        if args.save_checkpoints:
            baseline_checkpoint_path = logger.checkpoint_dir / "checkpoint_baseline.pt"
            torch.save({
                'step': 0,
                'model_state_dict': model.state_dict(),
                'checkpoint_type': 'baseline',
                'description': 'Initial model state before any modifications'
            }, baseline_checkpoint_path)
            # Also save using logger for consistency
            logger.save_checkpoint(
                step=0,
                model=model,
                additional_state={
                    'checkpoint_type': 'baseline',
                    'description': 'Initial model state before any modifications'
                }
            )
            vprint("Baseline model checkpoint saved to disk", 'quiet', verbosity)
        else:
            # Keep baseline in memory so we can reset before treatment when control group runs
            baseline_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 1. Baseline measurement
    vprint("\n" + "=" * 50, 'normal', verbosity)
    vprint("PHASE 1: Baseline Measurement", 'normal', verbosity)
    vprint("=" * 50, 'normal', verbosity)
    baseline = measure_baseline(model, tokenizer, args.device, verbosity=verbosity)
    logger.log_metrics(0, baseline, prefix="baseline")
    vprint(
        f"Baseline - Perplexity: {baseline['perplexity']:.2f}, "
        f"Effective Rank: {baseline['effective_rank']:.2f}",
        'normal',
        verbosity,
    )

    # 2. Control group (optional)
    if not args.skip_control:
        # Reset to baseline before control group (only if we have a saved or in-memory baseline)
        baseline_checkpoint_path = logger.checkpoint_dir / "checkpoint_baseline.pt"
        if baseline_checkpoint_path.exists():
            logger.load_checkpoint(
                checkpoint_path=str(baseline_checkpoint_path),
                model=model
            )
            vprint("Model reset to baseline state before control group", 'quiet', verbosity)
        elif baseline_state_dict is not None:
            # Restore from in-memory baseline (--save-checkpoints was False)
            model.load_state_dict({k: v.to(args.device) for k, v in baseline_state_dict.items()})
            vprint("Model reset to baseline state before control group (from memory)", 'quiet', verbosity)

        vprint("\n" + "=" * 50, 'normal', verbosity)
        vprint("PHASE 2: Control Group (Random Noise)", 'normal', verbosity)
        vprint("=" * 50, 'normal', verbosity)
        control_history = run_control_group(
            model,
            tokenizer,
            args.device,
            num_steps=config.attack.num_steps,
            learning_rate=config.attack.learning_rate,
            verbosity=verbosity,
        )

        for entry in control_history:
            logger.log_metrics(entry['iteration'], entry, prefix="control")

        # Plot control results
        plot_rank_collapse(
            control_history,
            save_path=str(logger.get_log_path() / "control_rank_collapse.png")
        )

        # Save checkpoint after control group only if --save-checkpoints
        if args.save_checkpoints:
            post_control_checkpoint_path = logger.checkpoint_dir / "checkpoint_post_control.pt"
            torch.save({
                'step': config.attack.num_steps,
                'model_state_dict': model.state_dict(),
                'checkpoint_type': 'post_control',
                'description': 'Model state after control group (random noise)'
            }, post_control_checkpoint_path)
            logger.save_checkpoint(
                step=config.attack.num_steps,
                model=model,
                additional_state={
                    'checkpoint_type': 'post_control',
                    'description': 'Model state after control group (random noise)'
                }
            )
            vprint("Post-control checkpoint saved", 'quiet', verbosity)

    # 3. Treatment group
    vprint("\n" + "=" * 50, 'normal', verbosity)
    vprint("PHASE 3: Treatment Group (Eigen-Prion)", 'normal', verbosity)
    vprint("=" * 50, 'normal', verbosity)

    # Always reset to baseline before treatment group
    baseline_checkpoint_path = logger.checkpoint_dir / "checkpoint_baseline.pt"
    if baseline_checkpoint_path.exists():
        logger.load_checkpoint(
            checkpoint_path=str(baseline_checkpoint_path),
            model=model
        )
        vprint("Model reset to baseline state before treatment group", 'quiet', verbosity)
    elif baseline_state_dict is not None:
        # Restore from in-memory baseline (--save-checkpoints was False)
        model.load_state_dict({k: v.to(args.device) for k, v in baseline_state_dict.items()})
        vprint("Model reset to baseline state before treatment group (from memory)", 'quiet', verbosity)
    else:
        vprint(
            "Warning: No baseline checkpoint (file or memory); treatment runs on current model state.",
            'normal',
            verbosity,
        )

    treatment = run_treatment_group(model, tokenizer, args.device, config, verbosity=verbosity)

    # Log treatment results
    for entry in treatment['history']:
        logger.log_metrics(entry['iteration'], entry, prefix="treatment")

    logger.log_metrics(
        config.attack.num_steps,
        treatment['results'],
        prefix="final"
    )

    # Plot treatment results
    plot_rank_collapse(
        treatment['history'],
        save_path=str(logger.get_log_path() / "treatment_rank_collapse.png")
    )
    plot_rank_reduction(
        treatment['history'],
        save_path=str(logger.get_log_path() / "treatment_rank_reduction.png")
    )

    # Save checkpoint after treatment group only if --save-checkpoints
    if args.save_checkpoints:
        post_treatment_checkpoint_path = logger.checkpoint_dir / "checkpoint_post_treatment.pt"
        torch.save({
            'step': config.attack.num_steps,
            'model_state_dict': model.state_dict(),
            'checkpoint_type': 'post_treatment',
            'description': 'Model state after treatment group (Eigen-Prion attack)'
        }, post_treatment_checkpoint_path)
        logger.save_checkpoint(
            step=config.attack.num_steps,
            model=model,
            additional_state={
                'checkpoint_type': 'post_treatment',
                'description': 'Model state after treatment group (Eigen-Prion attack)'
            }
        )
        vprint("Post-treatment checkpoint saved", 'quiet', verbosity)

    # 4. Post-attack measurement
    vprint("\n" + "=" * 50, 'normal', verbosity)
    vprint("PHASE 4: Post-Attack Measurement", 'normal', verbosity)
    vprint("=" * 50, 'normal', verbosity)
    post_attack = measure_baseline(model, tokenizer, args.device, verbosity=verbosity)
    logger.log_metrics(config.attack.num_steps + 1, post_attack, prefix="post_attack")

    vprint(
        f"Post-Attack - Perplexity: {post_attack['perplexity']:.2f}, "
        f"Effective Rank: {post_attack['effective_rank']:.2f}",
        'normal',
        verbosity,
    )

    # 5. Generate Spectral Decay Plot (The "Flag" Image)
    vprint("\n" + "=" * 50, 'normal', verbosity)
    vprint("PHASE 5: Generating Spectral Decay Plot", 'normal', verbosity)
    vprint("=" * 50, 'normal', verbosity)
    if len(baseline.get('singular_values', [])) > 0 and len(post_attack.get('singular_values', [])) > 0:
        plot_spectral_decay(
            baseline['singular_values'],
            post_attack['singular_values'],
            save_path=str(logger.get_log_path() / "spectral_decay.png"),
            title=f"Spectral Decay: {config.model.name} - Healthy vs Fibrotic"
        )
        vprint("Spectral decay plot generated successfully!", 'quiet', verbosity)
    else:
        vprint(
            "Warning: Could not generate spectral decay plot - missing singular values",
            'quiet',
            verbosity,
        )

    # Summary
    vprint("\n" + "=" * 50, 'normal', verbosity)
    vprint("EXPERIMENT SUMMARY", 'normal', verbosity)
    vprint("=" * 50, 'normal', verbosity)
    vprint(f"Initial Effective Rank: {baseline['effective_rank']:.2f}", 'normal', verbosity)
    vprint(f"Final Effective Rank: {post_attack['effective_rank']:.2f}", 'normal', verbosity)

    rank_reduction = 0.0
    if baseline['effective_rank'] > 0:
        rank_reduction = 1.0 - (post_attack['effective_rank'] / baseline['effective_rank'])
        vprint(f"Rank Reduction: {rank_reduction*100:.2f}%", 'normal', verbosity)

    vprint(f"\nResults saved to: {logger.get_log_path()}", 'quiet', verbosity)

    return {
        'run_number': run_number,
        'baseline_perplexity': baseline['perplexity'],
        'baseline_rank': baseline['effective_rank'],
        'post_attack_perplexity': post_attack['perplexity'],
        'post_attack_rank': post_attack['effective_rank'],
        'rank_reduction': rank_reduction,
        'log_path': str(logger.get_log_path())
    }


def display_aggregate_results(results: list, verbosity: str):
    """Display aggregate statistics across multiple runs."""
    if not results:
        vprint("No results to aggregate", 'quiet', verbosity)
        return

    vprint(f"\n{'='*50}", 'quiet', verbosity)
    vprint("AGGREGATE RESULTS", 'quiet', verbosity)
    vprint(f"{'='*50}", 'quiet', verbosity)

    # Extract metrics
    rank_reductions = [r['rank_reduction'] * 100 for r in results]
    baseline_ranks = [r['baseline_rank'] for r in results]
    post_ranks = [r['post_attack_rank'] for r in results]

    # Compute statistics
    import statistics
    vprint(f"Total runs: {len(results)}", 'quiet', verbosity)
    vprint(f"\nRank Reduction:", 'quiet', verbosity)
    vprint(f"  Mean: {statistics.mean(rank_reductions):.2f}%", 'quiet', verbosity)
    if len(rank_reductions) > 1:
        vprint(f"  Std Dev: {statistics.stdev(rank_reductions):.2f}%", 'quiet', verbosity)
        vprint(f"  Min: {min(rank_reductions):.2f}%", 'quiet', verbosity)
        vprint(f"  Max: {max(rank_reductions):.2f}%", 'quiet', verbosity)

    vprint(f"\nBaseline Effective Rank:", 'quiet', verbosity)
    vprint(f"  Mean: {statistics.mean(baseline_ranks):.2f}", 'quiet', verbosity)

    vprint(f"\nPost-Attack Effective Rank:", 'quiet', verbosity)
    vprint(f"  Mean: {statistics.mean(post_ranks):.2f}", 'quiet', verbosity)

    # Save aggregate results to CSV (under results dir, one file per model/quant)
    results_dir = Path(results[0]['log_path']).parent
    base_name = re.sub(r'_run\d+', '', Path(results[0]['log_path']).name)
    summary_file = results_dir / f"aggregate_summary_{base_name}.csv"
    with open(summary_file, 'w') as f:
        f.write("run,baseline_rank,post_attack_rank,rank_reduction_pct\n")
        for r in results:
            f.write(f"{r['run_number']},{r['baseline_rank']:.4f},{r['post_attack_rank']:.4f},{r['rank_reduction']*100:.2f}\n")

    vprint(f"\nAggregate summary saved to: {summary_file}", 'quiet', verbosity)


def main():
    parser = argparse.ArgumentParser(description="Run metabolic attack experiment")
    parser.add_argument(
        '--model',
        type=str,
        choices=['70m', '160m', '410m', '1b', '1.4b', '2.8b'],
        default='160m',
        help='Model size to use'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Computing device'
    )
    parser.add_argument(
        '--skip-control',
        action='store_true',
        help='Skip control group experiment'
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume from (optional)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['none', 'fp16', '4bit', '8bit'],
        default=None,
        help='Quantization mode (overrides config). "none"/"fp16" for full precision, "4bit" or "8bit" for quantized'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of times to repeat the experiment (for statistical analysis)'
    )
    parser.add_argument(
        '--verbosity',
        type=str,
        choices=['quiet', 'normal', 'verbose'],
        default='normal',
        help='Output verbosity level. "quiet" for minimal output (useful for batch runs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides config)'
    )
    parser.add_argument(
        '--save-checkpoints',
        action='store_true',
        default=False,
        help='Save checkpoint files to disk (default: False, saves disk space)'
    )
    parser.add_argument(
        '--force-fft',
        action='store_true',
        help='Force Full Fine-Tuning mode (disable LoRA, enable gradient checkpointing)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        if args.model == '70m':
            config = get_pythia_70m_config()
        elif args.model == '160m':
            config = get_pythia_160m_config()
        elif args.model == '410m':
            config = get_pythia_410m_config()
        elif args.model == '1b':
            config = get_pythia_1b_config()
        elif args.model == '1.4b':
            config = get_pythia_1_4b_config()
        elif args.model == '2.8b':
            config = get_pythia_2_8b_config()
        else:
            raise ValueError(f"Unknown model: {args.model}")
    
    config.device = args.device
    config.output_dir = args.output_dir
    
    # Apply command-line overrides
    if args.quantization:
        if args.quantization in ['none', 'fp16']:
            config.model.use_quantization = False
        elif args.quantization == '4bit':
            config.model.use_quantization = True
            config.model.quantization_bits = 4
        elif args.quantization == '8bit':
            config.model.use_quantization = True
            config.model.quantization_bits = 8
    
    if args.seed is not None:
        config.seed = args.seed
    
    if args.force_fft:
        config.model.use_fft = True
        config.model.use_gradient_checkpointing = True

    # Storage for multi-run results
    all_results = []

    vprint(f"\n{'='*50}", 'normal', args.verbosity)
    vprint(f"EXPERIMENT CONFIGURATION", 'normal', args.verbosity)
    vprint(f"{'='*50}", 'normal', args.verbosity)
    vprint(f"Model: {args.model}", 'normal', args.verbosity)
    vprint(f"Quantization: {args.quantization or 'from config'}", 'normal', args.verbosity)
    vprint(f"Number of runs: {args.num_runs}", 'normal', args.verbosity)
    vprint(f"Seed: {config.seed}", 'normal', args.verbosity)
    vprint(f"Save checkpoints: {args.save_checkpoints}", 'normal', args.verbosity)

    # Run experiments
    for run_num in range(1, args.num_runs + 1):
        actual_seed = config.seed + run_num
        vprint(f"\n{'#'*50}", 'normal', args.verbosity)
        vprint(f"RUN {run_num}/{args.num_runs}", 'normal', args.verbosity)
        vprint(f"Using seed: {actual_seed} (base_seed={config.seed} + run_number={run_num})", 'normal', args.verbosity)
        vprint(f"{'#'*50}", 'normal', args.verbosity)

        try:
            result = run_single_experiment(config, args, run_num, args.verbosity)
            all_results.append(result)

            # Print run summary
            vprint(f"\nRun {run_num} Summary:", 'normal', args.verbosity)
            vprint(f"  Rank Reduction: {result['rank_reduction']*100:.2f}%", 'normal', args.verbosity)
            vprint(f"  Results saved to: {result['log_path']}", 'quiet', args.verbosity)

        except Exception as e:
            vprint(f"\nERROR in run {run_num}: {e}", 'quiet', args.verbosity)
            if args.verbosity == 'verbose':
                import traceback
                traceback.print_exc()
            continue

    # Aggregate and display results
    if args.num_runs > 1:
        display_aggregate_results(all_results, args.verbosity)


if __name__ == "__main__":
    main()
