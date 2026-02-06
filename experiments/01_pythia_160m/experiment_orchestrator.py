#!/usr/bin/env python3
"""
Experiment Orchestrator for Roadmap Priorities

Coordinates multi-model experiments according to roadmap_#2.md
Uses run_experiment.py as infrastructure.

This script orchestrates:
- Priority 1: Scaling Law Curve (70m, 160m, 410m, 1b models)
- Priority 2: Placebo Test (410m with gaussian_noise, random_text, eigen_prion)
- Priority 3: Optimizer Comparison (410m with AdamW vs SGD)
- Priority 4: Quantization Shield (1.4b/410m with FP16 vs 8-bit vs 4-bit)
"""

import subprocess
import argparse
import sys
from pathlib import Path
from typing import List, Optional


def run_experiment_command(
    model: str,
    control_type: str = 'eigen_prion',
    num_runs: int = 1,
    output_dir: str = './experiments/results',
    quantization: str = 'fp16',
    seed: int = 42,
    verbosity: str = 'normal',
    force_fft: bool = True,
    skip_control: bool = False,
    optimizer: str = 'adamw',
    learning_rate: Optional[float] = None
) -> subprocess.CompletedProcess:
    """
    Run a single experiment by calling run_experiment.py.
    
    Args:
        model: Model size (70m, 160m, 410m, 1b, etc.)
        control_type: Type of control/treatment (none, random_tokens, gaussian_noise, random_text, eigen_prion)
        num_runs: Number of runs to execute
        output_dir: Base output directory
        quantization: Quantization mode (fp16, 4bit, 8bit)
        seed: Random seed
        verbosity: Verbosity level (quiet, normal, verbose)
        force_fft: Whether to force Full Fine-Tuning mode
        skip_control: Whether to skip control (deprecated, use control_type='none')
        optimizer: Optimizer to use ('adamw' or 'sgd')
        learning_rate: Learning rate override (None to use config default)
        
    Returns:
        CompletedProcess: Result of subprocess.run()
    """
    script_path = Path(__file__).parent / "run_experiment.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        '--model', model,
        '--control-type', control_type,
        '--num-runs', str(num_runs),
        '--output-dir', output_dir,
        '--quantization', quantization,
        '--seed', str(seed),
        '--verbosity', verbosity,
        '--optimizer', optimizer,
    ]
    
    if force_fft:
        cmd.append('--force-fft')
    
    if skip_control:
        cmd.append('--skip-control')
    
    if learning_rate is not None:
        cmd.extend(['--learning-rate', str(learning_rate)])
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result


def run_priority_1(args):
    """
    Priority 1: Scaling Law Curve
    
    Models: 70m, 160m, 410m, 1b
    Config: FP16, FFT, 10 runs each (or as specified)
    Note: Skips control group (uses --skip-control) 
    """
    print("\n" + "="*70)
    print("PRIORITY 1: Scaling Law Curve")
    print("="*70)
    print(f"Models: 70m, 160m, 410m, 1b")
    print(f"Config: FP16, FFT, {args.num_runs} runs each")
    print(f"Control group: SKIPPED (runs treatment only)")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # models = ['70m', '160m', '410m', '1b']  # default
    models = ['1b']  # Used for debugging
    results = []
    
    for model in models:
        print(f"\n{'#'*70}")
        print(f"Processing model: {model}")
        print(f"{'#'*70}\n")
        
        result = run_experiment_command(
            model=model,
            control_type='eigen_prion',  # Treatment group only 
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            quantization='fp16',
            seed=args.seed,
            verbosity=args.verbosity,
            force_fft=True,
            skip_control=True,  # Skip control group phase for Priority 1 
            learning_rate=args.learning_rate
        )
        
        results.append({
            'model': model,
            'returncode': result.returncode,
            'success': result.returncode == 0
        })
        
        if result.returncode != 0:
            print(f"\nWARNING: Model {model} experiment failed with return code {result.returncode}")
        else:
            print(f"\nSUCCESS: Model {model} experiment completed")
    
    # Summary
    print("\n" + "="*70)
    print("PRIORITY 1 SUMMARY")
    print("="*70)
    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        print(f"  {r['model']:6s}: {status}")
    print("="*70 + "\n")
    
    return results


def run_priority_2(args):
    """
    Priority 2: Placebo Test (Specificity)
    
    Model: 410m
    Groups: gaussian_noise, random_text, eigen_prion
    Runs: 3 each (or as specified)
    """
    print("\n" + "="*70)
    print("PRIORITY 2: Placebo Test (Specificity)")
    print("="*70)
    print(f"Model: 410m")
    print(f"Control Types: gaussian_noise, random_text, eigen_prion")
    print(f"Runs per type: {args.num_runs}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    control_types = ['gaussian_noise', 'random_text', 'eigen_prion']
    results = []
    
    for control_type in control_types:
        print(f"\n{'#'*70}")
        print(f"Processing control type: {control_type}")
        print(f"{'#'*70}\n")
        
        result = run_experiment_command(
            model='410m',
            control_type=control_type,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            quantization='fp16',
            seed=args.seed,
            verbosity=args.verbosity,
            force_fft=True,
            learning_rate=args.learning_rate
        )
        
        results.append({
            'control_type': control_type,
            'returncode': result.returncode,
            'success': result.returncode == 0
        })
        
        if result.returncode != 0:
            print(f"\nWARNING: Control type {control_type} experiment failed with return code {result.returncode}")
        else:
            print(f"\nSUCCESS: Control type {control_type} experiment completed")
    
    # Summary
    print("\n" + "="*70)
    print("PRIORITY 2 SUMMARY")
    print("="*70)
    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        print(f"  {r['control_type']:20s}: {status}")
    print("="*70 + "\n")
    
    return results


def run_priority_3(args):
    """
    Priority 3: Optimizer Comparison (Mechanism Test)
    
    Model: 410m (FP16, FFT)
    Optimizers: AdamW vs SGD (no momentum)
    Runs: 3 each
    Expected: SGD immune, AdamW vulnerable
    """
    print("\n" + "="*70)
    print("PRIORITY 3: Optimizer Comparison (Mechanism Test)")
    print("="*70)
    print(f"Model: 410m")
    print(f"Optimizers: AdamW vs SGD (no momentum)")
    print(f"Runs per optimizer: {args.num_runs}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    optimizers = ['adamw', 'sgd']
    results = []
    
    for optimizer in optimizers:
        print(f"\n{'#'*70}")
        print(f"Processing optimizer: {optimizer.upper()}")
        print(f"{'#'*70}\n")
        
        result = run_experiment_command(
            model='410m',
            control_type='eigen_prion',  # Treatment group
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            quantization='fp16',
            seed=args.seed,
            verbosity=args.verbosity,
            force_fft=True,
            skip_control=True,  # Skip control group for Priority 3
            optimizer=optimizer,  # Specify optimizer
            learning_rate=args.learning_rate
        )
        
        results.append({
            'optimizer': optimizer,
            'returncode': result.returncode,
            'success': result.returncode == 0
        })
        
        if result.returncode != 0:
            print(f"\nWARNING: Optimizer {optimizer} experiment failed with return code {result.returncode}")
        else:
            print(f"\nSUCCESS: Optimizer {optimizer} experiment completed")
    
    # Summary
    print("\n" + "="*70)
    print("PRIORITY 3 SUMMARY")
    print("="*70)
    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        print(f"  {r['optimizer'].upper():6s}: {status}")
    print("="*70 + "\n")
    
    return results


def run_priority_4(args):
    """
    Priority 4: Quantization Shield (Shield Matrix)
    
    Model: Configurable via --model (default: 1b, supports: 70m, 160m, 410m, 1b, 1.4b, 2.8b)
    Quantization: FP16 vs 8-bit vs 4-bit
    Runs: 3 each (or as specified)
    Expected: FP16 collapses > 8-bit slight damage/immune > 4-bit completely immune
    """
    # Get model from args, default to 1b
    model = getattr(args, 'model', '1b')
    
    print("\n" + "="*70)
    print("PRIORITY 4: Quantization Shield (Shield Matrix)")
    print("="*70)
    print(f"Model: {model}")
    print(f"Quantization levels: FP16, 8-bit, 4-bit")
    print(f"Runs per quantization: {args.num_runs}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    quantization_levels = ['fp16', '8bit', '4bit']  # default
    # quantization_levels = ['4bit']  # used for debugging
    results = []
    
    for quantization in quantization_levels:
        print(f"\n{'#'*70}")
        print(f"Processing quantization: {quantization.upper()}")
        print(f"{'#'*70}\n")
        
        # FFT handling: FP16 needs FFT, quantized models don't (quantization already reduces memory)
        force_fft = (quantization == 'fp16')
        
        result = run_experiment_command(
            model=model,
            control_type='eigen_prion',  # Treatment group
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            quantization=quantization,
            seed=args.seed,
            verbosity=args.verbosity,
            force_fft=force_fft,
            skip_control=True,  # Skip control group for Priority 4
            optimizer='adamw',  # Use AdamW (default)
            learning_rate=args.learning_rate
        )
        
        results.append({
            'quantization': quantization,
            'returncode': result.returncode,
            'success': result.returncode == 0
        })
        
        if result.returncode != 0:
            print(f"\nWARNING: Quantization {quantization} experiment failed with return code {result.returncode}")
        else:
            print(f"\nSUCCESS: Quantization {quantization} experiment completed")
    
    # Summary
    print("\n" + "="*70)
    print("PRIORITY 4 SUMMARY")
    print("="*70)
    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        print(f"  {r['quantization'].upper():6s}: {status}")
    print("="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment Orchestrator for Roadmap Priorities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Priority 1 (Scaling Law) - uses default output dir: ./experiments/results/priority1_scaling_law/raw_data/
  python experiment_orchestrator.py --priority 1 --num-runs 10
  
  # Run Priority 2 (Placebo Test) - uses default output dir: ./experiments/results/priority2_placebo/raw_data/
  python experiment_orchestrator.py --priority 2 --num-runs 3
  
  # Run Priority 3 (Optimizer Comparison) - uses default output dir: ./experiments/results/priority3_mechanism/raw_data/
  python experiment_orchestrator.py --priority 3 --num-runs 3
  
  # Run Priority 4 (Quantization Shield) with default 1b model - uses default output dir: ./experiments/results/priority4_shield/raw_data/
  python experiment_orchestrator.py --priority 4
  
  # Run Priority 4 with different models (1.4b, 410m, etc.)
  python experiment_orchestrator.py --priority 4 --model 1.4b
  python experiment_orchestrator.py --priority 4 --model 410m
  
  # Run with custom output directory (overrides default)
  python experiment_orchestrator.py --priority 1 --output-dir ./custom/results/
        """
    )
    
    parser.add_argument(
        '--priority',
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help='Priority level to run (1=Scaling Law, 2=Placebo Test, 3=Optimizer, 4=Quantization)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,  # Will be set based on priority if not provided
        help='Base output directory for results (default: priority-specific directory)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Number of runs per experiment (default: 10 for Priority 1, 3 for Priorities 2-4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbosity',
        type=str,
        choices=['quiet', 'normal', 'verbose'],
        default='normal',
        help='Output verbosity level'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['70m', '160m', '410m', '1b', '1.4b', '2.8b'],
        default='1b',
        help='Model to use (default: 1b for Priority 4, used by Priority 4; other priorities use fixed models)'
    )
    parser.add_argument(
        '--learning-rate',
        '--lr',
        type=float,
        default=None,
        help='Override learning rate for all experiments in this priority'
    )
    
    args = parser.parse_args()
    
    # Set default output directory based on priority if not provided
    if args.output_dir is None:
        priority_dirs = {
            1: './experiments/results/priority1_scaling_law/raw_data/',
            2: './experiments/results/priority2_placebo/raw_data/',
            3: './experiments/results/priority3_mechanism/raw_data/',
            4: './experiments/results/priority4_shield/raw_data/',
        }
        args.output_dir = priority_dirs[args.priority]
        print(f"Using default output directory for Priority {args.priority}: {args.output_dir}")
    
    # Adjust default num_runs based on priority
    if args.priority == 2 and args.num_runs == 10:
        # Priority 2 typically uses 3 runs
        args.num_runs = 3
        print("Note: Priority 2 typically uses 3 runs. Using 3 runs.")
    elif args.priority == 3 and args.num_runs == 10:
        # Priority 3 typically uses 3 runs
        args.num_runs = 3
        print("Note: Priority 3 typically uses 3 runs. Using 3 runs.")
    elif args.priority == 4 and args.num_runs == 10:
        # Priority 4 typically uses 3 runs
        args.num_runs = 3
        print("Note: Priority 4 typically uses 3 runs. Using 3 runs.")
    
    # Route to appropriate priority function
    if args.priority == 1:
        run_priority_1(args)
    elif args.priority == 2:
        run_priority_2(args)
    elif args.priority == 3:
        run_priority_3(args)
    elif args.priority == 4:
        run_priority_4(args)
    else:
        print(f"Error: Unknown priority {args.priority}")
        sys.exit(1)


if __name__ == "__main__":
    main()
