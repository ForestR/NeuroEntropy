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
from src.config import ExperimentConfig, get_pythia_160m_config, get_pythia_70m_config, get_pythia_410m_config
from src.utils.logger import ExperimentLogger
from src.utils.visualizer import plot_rank_collapse, plot_rank_reduction, plot_spectral_decay
from experiments.utils.evaluate import compute_perplexity, create_simple_test_set


def load_model(config: ExperimentConfig, device: str):
    """Load model and tokenizer."""
    print(f"Loading model: {config.model.hf_model_id}")
    
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
    
    model.eval()
    return model, tokenizer


def measure_baseline(
    model,
    tokenizer,
    device: str,
    num_samples: int = 10
) -> dict:
    """Measure baseline metrics."""
    print("Measuring baseline metrics...")
    
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
        print(f"Warning: Could not measure effective rank: {e}")
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
    learning_rate: float = 1e-4
) -> list:
    """Run control group: inject random noise."""
    print(f"Running control group: {num_steps} steps of random noise...")
    
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
    config: ExperimentConfig
) -> dict:
    """Run treatment group: inject Eigen-Prion catalyst."""
    print(f"Running treatment group: {config.attack.num_steps} steps of Eigen-Prion...")
    
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
        learning_rate=config.attack.learning_rate
    )
    
    return {
        'results': results,
        'history': attack_loop.get_history()
    }


def main():
    parser = argparse.ArgumentParser(description="Run metabolic attack experiment")
    parser.add_argument(
        '--model',
        type=str,
        choices=['70m', '160m', '410m'],
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
        else:
            raise ValueError(f"Unknown model: {args.model}")
    
    config.device = args.device
    config.output_dir = args.output_dir
    
    # Initialize logger
    experiment_name = f"pythia_{args.model}_{config.model.hf_model_id.split('/')[-1]}"
    logger = ExperimentLogger(
        output_dir=config.output_dir,
        experiment_name=experiment_name
    )
    
    logger.log_config({
        'model': config.model.__dict__,
        'attack': config.attack.__dict__,
        'hvp': config.hvp.__dict__,
    })
    
    # Load model
    model, tokenizer = load_model(config, args.device)
    
    # Resume from checkpoint if specified, otherwise save baseline checkpoint
    if args.resume_from_checkpoint:
        logger.load_checkpoint(
            checkpoint_path=args.resume_from_checkpoint,
            model=model
        )
        print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
    else:
        # Always save initial model state (baseline checkpoint)
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
        print("Baseline model checkpoint saved to disk")
    
    # 1. Baseline measurement
    print("\n" + "="*50)
    print("PHASE 1: Baseline Measurement")
    print("="*50)
    baseline = measure_baseline(model, tokenizer, args.device)
    logger.log_metrics(0, baseline, prefix="baseline")
    print(f"Baseline - Perplexity: {baseline['perplexity']:.2f}, "
          f"Effective Rank: {baseline['effective_rank']:.2f}")
    
    # 2. Control group (optional)
    if not args.skip_control:
        # Reset to baseline before control group
        baseline_checkpoint_path = logger.checkpoint_dir / "checkpoint_baseline.pt"
        if baseline_checkpoint_path.exists():
            logger.load_checkpoint(
                checkpoint_path=str(baseline_checkpoint_path),
                model=model
            )
            print("Model reset to baseline state before control group")
        
        print("\n" + "="*50)
        print("PHASE 2: Control Group (Random Noise)")
        print("="*50)
        control_history = run_control_group(
            model,
            tokenizer,
            args.device,
            num_steps=config.attack.num_steps,
            learning_rate=config.attack.learning_rate
        )
        
        for entry in control_history:
            logger.log_metrics(entry['iteration'], entry, prefix="control")
        
        # Plot control results
        plot_rank_collapse(
            control_history,
            save_path=str(logger.get_log_path() / "control_rank_collapse.png")
        )
        
        # Save checkpoint after control group
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
        print("Post-control checkpoint saved")
    
    # 3. Treatment group
    print("\n" + "="*50)
    print("PHASE 3: Treatment Group (Eigen-Prion)")
    print("="*50)
    
    # Always reset to baseline before treatment group
    baseline_checkpoint_path = logger.checkpoint_dir / "checkpoint_baseline.pt"
    if baseline_checkpoint_path.exists():
        logger.load_checkpoint(
            checkpoint_path=str(baseline_checkpoint_path),
            model=model
        )
        print("Model reset to baseline state before treatment group")
    
    treatment = run_treatment_group(model, tokenizer, args.device, config)
    
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
    
    # Save checkpoint after treatment group
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
    print("Post-treatment checkpoint saved")
    
    # 4. Post-attack measurement
    print("\n" + "="*50)
    print("PHASE 4: Post-Attack Measurement")
    print("="*50)
    post_attack = measure_baseline(model, tokenizer, args.device)
    logger.log_metrics(config.attack.num_steps + 1, post_attack, prefix="post_attack")
    
    print(f"Post-Attack - Perplexity: {post_attack['perplexity']:.2f}, "
          f"Effective Rank: {post_attack['effective_rank']:.2f}")
    
    # 5. Generate Spectral Decay Plot (The "Flag" Image)
    print("\n" + "="*50)
    print("PHASE 5: Generating Spectral Decay Plot")
    print("="*50)
    if len(baseline.get('singular_values', [])) > 0 and len(post_attack.get('singular_values', [])) > 0:
        plot_spectral_decay(
            baseline['singular_values'],
            post_attack['singular_values'],
            save_path=str(logger.get_log_path() / "spectral_decay.png"),
            title=f"Spectral Decay: {config.model.name} - Healthy vs Fibrotic"
        )
        print("Spectral decay plot generated successfully!")
    else:
        print("Warning: Could not generate spectral decay plot - missing singular values")
    
    # Summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Initial Effective Rank: {baseline['effective_rank']:.2f}")
    print(f"Final Effective Rank: {post_attack['effective_rank']:.2f}")
    if baseline['effective_rank'] > 0:
        rank_reduction = 1.0 - (post_attack['effective_rank'] / baseline['effective_rank'])
        print(f"Rank Reduction: {rank_reduction*100:.2f}%")
    
    print(f"\nResults saved to: {logger.get_log_path()}")


if __name__ == "__main__":
    main()
