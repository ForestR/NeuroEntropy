#!/usr/bin/env python3
"""
Scaling Law Comparison Script

Runs metabolic attack experiments across multiple model sizes to validate
the scaling law hypothesis: larger models are more vulnerable to attacks.
"""

import sys
import os
import torch
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import (
    ExperimentConfig,
    get_pythia_70m_config,
    get_pythia_160m_config,
    get_pythia_410m_config
)
from experiments.utils.evaluate import compute_perplexity, create_simple_test_set
from src.diagnosis import ActivationHook, compute_effective_rank, extract_singular_values
from src.utils.visualizer import plot_scaling_law, plot_rank_collapse_heatmap, plot_spectral_decay


def run_experiment_on_model(
    model_name: str,
    config: ExperimentConfig,
    num_steps: int = 100,
    device: str = "cuda"
) -> Dict:
    """
    Run attack experiment on a single model.
    
    Args:
        model_name: Name identifier for the model
        config: Experiment configuration
        num_steps: Number of attack steps
        device: Computing device
        
    Returns:
        results: Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment on {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading {config.model.hf_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if config.model.use_quantization:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True if config.model.quantization_bits == 4 else False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model.hf_model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        device = "cuda:0"  # Quantized models use device_map
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.hf_model_id
        ).to(device)
    
    model.eval()
    
    # Measure initial rank and singular values
    print("Measuring initial effective rank and singular values...")
    initial_rank, initial_sv = measure_effective_rank(model, tokenizer, device, return_singular_values=True)
    print(f"Initial effective rank: {initial_rank:.2f}")
    
    # Run attack (simplified - in practice would use full attack loop)
    print(f"Running attack for {num_steps} steps...")
    attack_history = run_simplified_attack(
        model,
        tokenizer,
        device,
        num_steps=num_steps,
        learning_rate=config.attack.learning_rate
    )
    
    # Measure final rank and singular values
    print("Measuring final effective rank and singular values...")
    final_rank, final_sv = measure_effective_rank(model, tokenizer, device, return_singular_values=True)
    print(f"Final effective rank: {final_rank:.2f}")
    
    # Compute rank reduction
    if initial_rank > 0:
        rank_reduction = 1.0 - (final_rank / initial_rank)
    else:
        rank_reduction = 0.0
    
    print(f"Rank reduction: {rank_reduction*100:.2f}%")
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'model_name': model_name,
        'model_size': get_model_size(config.model.hf_model_id),
        'initial_rank': initial_rank,
        'final_rank': final_rank,
        'rank_reduction': rank_reduction,
        'rank_reduction_pct': rank_reduction * 100,
        'history': attack_history,
        'singular_values_before': initial_sv.tolist() if initial_sv is not None else [],
        'singular_values_after': final_sv.tolist() if final_sv is not None else []
    }


def measure_effective_rank(
    model,
    tokenizer,
    device: str,
    num_samples: int = 5,
    return_singular_values: bool = False
):
    """
    Measure effective rank of model activations.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        device: Computing device
        num_samples: Number of samples to collect
        return_singular_values: If True, also return singular values
        
    Returns:
        effective_rank: Effective rank value
        singular_values: (optional) Array of singular values if return_singular_values=True
    """
    hook = ActivationHook()
    try:
        hook.register(model)
        
        test_texts = create_simple_test_set(num_samples)
        model.eval()
        
        with torch.no_grad():
            for text in test_texts[:num_samples]:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).to(device)
                _ = model(**inputs)
        
        activations = hook.get_activations()
        hook.clear()
        
        if activations is not None and activations.numel() > 0:
            effective_rank = compute_effective_rank(activations)
            if return_singular_values:
                singular_values = extract_singular_values(activations, normalize=True)
                return effective_rank, singular_values
            else:
                return effective_rank
        else:
            if return_singular_values:
                return 0.0, np.array([])
            else:
                return 0.0
    except Exception as e:
        print(f"Warning: Could not measure effective rank: {e}")
        if return_singular_values:
            return 0.0, np.array([])
        else:
            return 0.0


def run_simplified_attack(
    model,
    tokenizer,
    device: str,
    num_steps: int = 100,
    learning_rate: float = 1e-4
) -> List[Dict]:
    """
    Run simplified attack (for scaling comparison).
    
    In practice, this would use the full MetabolicAttackLoop,
    but for scaling experiments we use a simplified version.
    """
    model.train()
    history = []
    vocab_size = len(tokenizer)
    
    # Generate catalyst-like tokens (simplified)
    # In practice, would use HessianAwareCatalyst
    catalyst_tokens = torch.randint(
        0, vocab_size,
        (1, 128),
        device=device
    )
    
    for step in range(num_steps):
        # Forward and backward pass
        labels = catalyst_tokens[:, 1:].contiguous()
        input_ids = catalyst_tokens[:, :-1].contiguous()
        
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
        if step % 20 == 0 or step == num_steps - 1:
            model.eval()
            rank = measure_effective_rank(model, tokenizer, device, num_samples=3, return_singular_values=False)
            history.append({
                'iteration': step,
                'effective_rank': rank
            })
            model.train()
    
    model.eval()
    return history


def get_model_size(model_id: str) -> float:
    """Get approximate model size in parameters."""
    size_map = {
        'pythia-70m': 70e6,
        'pythia-160m': 160e6,
        'pythia-410m': 410e6,
        'pythia-1b': 1e9,
        'pythia-1.4b': 1.4e9,
        'pythia-2.8b': 2.8e9,
        'pythia-6.9b': 6.9e9,
        'pythia-12b': 12e9,
    }
    
    for key, size in size_map.items():
        if key in model_id.lower():
            return size
    
    # Default estimate based on model ID
    if '70m' in model_id.lower():
        return 70e6
    elif '160m' in model_id.lower():
        return 160e6
    elif '410m' in model_id.lower():
        return 410e6
    elif '1b' in model_id.lower() or '1.4b' in model_id.lower():
        return 1.4e9
    elif '2.8b' in model_id.lower():
        return 2.8e9
    else:
        return 160e6  # Default


def main():
    parser = argparse.ArgumentParser(
        description="Compare scaling law across multiple model sizes"
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['70m', '160m', '410m'],
        default=['70m', '160m', '410m'],
        help='Model sizes to test'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=100,
        help='Number of attack steps'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments/results/scaling_law',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Computing device'
    )
    
    args = parser.parse_args()
    
    # Model configurations
    configs = {
        '70m': get_pythia_70m_config(),
        '160m': get_pythia_160m_config(),
        '410m': get_pythia_410m_config(),
    }
    
    # Run experiments
    all_results = []
    rank_history = {}
    
    for model_key in args.models:
        if model_key not in configs:
            print(f"Warning: Unknown model {model_key}, skipping")
            continue
        
        config = configs[model_key]
        config.device = args.device
        
        try:
            result = run_experiment_on_model(
                model_key,
                config,
                num_steps=args.num_steps,
                device=args.device
            )
            all_results.append(result)
            rank_history[model_key] = result['history']
        except Exception as e:
            print(f"Error running experiment on {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "scaling_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SCALING LAW RESULTS")
    print(f"{'='*60}")
    
    # Print summary
    for result in all_results:
        print(f"{result['model_name']:10s} | "
              f"Size: {result['model_size']/1e6:6.1f}M | "
              f"Rank Reduction: {result['rank_reduction_pct']:6.2f}%")
    
    # Generate spectral decay plots for each model
    print(f"\n{'='*60}")
    print("GENERATING SPECTRAL DECAY PLOTS")
    print(f"{'='*60}")
    for result in all_results:
        model_name = result['model_name']
        sv_before = result.get('singular_values_before', [])
        sv_after = result.get('singular_values_after', [])
        
        if len(sv_before) > 0 and len(sv_after) > 0:
            sv_before_array = np.array(sv_before)
            sv_after_array = np.array(sv_after)
            
            decay_plot_path = output_dir / f"spectral_decay_{model_name}.png"
            plot_spectral_decay(
                sv_before_array,
                sv_after_array,
                save_path=str(decay_plot_path),
                title=f"Spectral Decay: {model_name.upper()} - Healthy vs Fibrotic"
            )
            print(f"Spectral decay plot for {model_name} saved to: {decay_plot_path}")
        else:
            print(f"Warning: Missing singular values for {model_name}, skipping spectral decay plot")
    
    # Plot scaling law
    if len(all_results) >= 2:
        model_sizes = [r['model_size'] for r in all_results]
        rank_reductions = [r['rank_reduction_pct'] for r in all_results]
        
        plot_path = output_dir / "scaling_law.png"
        plot_scaling_law(
            model_sizes,
            rank_reductions,
            save_path=str(plot_path),
            title="Scaling Law: Model Vulnerability vs Size"
        )
        print(f"\nScaling law plot saved to: {plot_path}")
        
        # Plot heatmap
        if rank_history:
            iterations = []
            for history in rank_history.values():
                iterations.extend([h['iteration'] for h in history])
            iterations = sorted(list(set(iterations)))
            
            if iterations:
                heatmap_path = output_dir / "rank_collapse_heatmap.png"
                plot_rank_collapse_heatmap(
                    rank_history,
                    list(rank_history.keys()),
                    iterations,
                    save_path=str(heatmap_path)
                )
                print(f"Rank collapse heatmap saved to: {heatmap_path}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
