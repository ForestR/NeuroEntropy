#!/usr/bin/env python3
"""
Healing/Reversibility Experiment Runner

Tests whether spectral collapse induced by HAEP attacks can be reversed
through standard supervised fine-tuning (SFT) on clean data.

Protocol:
1. Load collapsed model (from checkpoint or run fresh attack)
2. Measure collapsed state metrics (rank, perplexity, spectral)
3. Run SFT on clean Pile data for ~500 steps
4. Track recovery trajectory (rank, perplexity every 10 steps)
5. Generate recovery visualizations and summary
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.catalyst import HessianAwareCatalyst
from src.attack_loop import MetabolicAttackLoop
from src.diagnosis import ActivationHook, compute_effective_rank, compute_spectral_gap, extract_singular_values
from src.config import (
    ExperimentConfig,
    HealingConfig,
    get_pythia_70m_config,
    get_pythia_160m_config,
    get_pythia_410m_config,
    get_pythia_1b_config,
    get_pythia_1_4b_config,
    get_pythia_2_8b_config,
)
from src.utils.logger import ExperimentLogger
from src.utils.visualizer import (
    plot_recovery_trajectory,
    plot_perplexity_recovery,
    plot_spectral_decay_three_state,
)
from experiments.utils.evaluate import compute_perplexity, create_simple_test_set, load_random_pile_texts


def vprint(message: str, level: str = "normal", verbosity: str = "normal"):
    """Verbosity-aware print function."""
    levels = {"quiet": 0, "normal": 1, "verbose": 2}
    if levels.get(verbosity, 1) >= levels.get(level, 1):
        print(message)


def discover_checkpoints(checkpoint_dir: str | Path, pattern: str = "checkpoint_post_treatment.pt") -> list[Path]:
    """
    Discover all checkpoint files matching the pattern under the given directory.
    Example: checkpoint_dir=.../raw_data -> finds .../pythia_1b_run01_fp16/checkpoints/checkpoint_post_treatment.pt
    """
    root = Path(checkpoint_dir)
    if not root.exists():
        return []
    found = list(root.rglob(pattern))
    return sorted(found)


def load_model(
    config: ExperimentConfig,
    device: str,
    verbosity: str = "normal",
    use_gradient_checkpointing: bool | None = None,
):
    """Load model and tokenizer.
    use_gradient_checkpointing: If False, disable gradient checkpointing (recommended for healing to reduce NaN).
    """
    vprint(f"Loading model: {config.model.hf_model_id}", "normal", verbosity)

    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.model.use_quantization:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.model.quantization_bits == 4,
            load_in_8bit=config.model.quantization_bits == 8,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model.hf_model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            model.config.use_cache = False
        peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            lora_dropout=config.lora.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model.hf_model_id).to(device)

    # Gradient checkpointing: use override if provided, else config (disable for healing to reduce NaN)
    gc_default = getattr(config.model, "use_gradient_checkpointing", False)
    gc_enable = gc_default if use_gradient_checkpointing is None else use_gradient_checkpointing
    if gc_enable and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif not gc_enable and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True

    if getattr(config.model, "use_fft", False) and not config.model.use_quantization:
        for param in model.parameters():
            param.requires_grad = True

    model.eval()
    return model, tokenizer


def measure_baseline(
    model,
    tokenizer,
    device: str,
    num_samples: int = 10,
    verbosity: str = "normal",
) -> dict:
    """Measure baseline metrics (perplexity, effective rank, spectral gap, singular values)."""
    vprint("Measuring metrics...", "normal", verbosity)

    test_texts = create_simple_test_set(num_samples)
    perplexity = compute_perplexity(
        model, tokenizer, test_texts, device, verbose=(verbosity != "quiet")
    )

    hook = ActivationHook()
    try:
        hook.register(model)
        model.eval()
        with torch.no_grad():
            for i in range(min(5, len(test_texts))):
                inputs = tokenizer(
                    test_texts[i],
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=True,  # Avoid heavy padding to reduce NaN on collapsed models
                ).to(device)
                _ = model(**inputs)

        activations = hook.get_activations()
        hook.clear()

        if activations is not None:
            effective_rank = compute_effective_rank(activations)
            spectral_gap = compute_spectral_gap(activations)
            singular_values = extract_singular_values(activations, normalize=True)
        else:
            effective_rank = 0.0
            spectral_gap = 0.0
            singular_values = np.array([])
    except Exception as e:
        vprint(f"Warning: Could not measure effective rank: {e}", "normal", verbosity)
        effective_rank = 0.0
        spectral_gap = 0.0
        singular_values = np.array([])

    return {
        "perplexity": perplexity,
        "effective_rank": effective_rank,
        "spectral_gap": spectral_gap,
        "singular_values": singular_values,
    }


def load_collapsed_model(
    checkpoint_path: str | None,
    config: ExperimentConfig,
    device: str,
    logger: ExperimentLogger,
    verbosity: str = "normal",
    no_gradient_checkpointing: bool = True,
):
    """
    Load collapsed model: from checkpoint if provided, otherwise run fresh HAEP attack.
    Returns (model, tokenizer, original_baseline, collapsed_metrics).
    no_gradient_checkpointing: Disable gradient checkpointing (recommended for healing to reduce NaN).
    """
    model, tokenizer = load_model(
        config, device, verbosity,
        use_gradient_checkpointing=False if no_gradient_checkpointing else None,
    )

    vprint("\nMeasuring original baseline (healthy state)...", "normal", verbosity)
    original_baseline = measure_baseline(model, tokenizer, device, verbosity=verbosity)

    if checkpoint_path and Path(checkpoint_path).exists():
        vprint(f"Loading collapsed model from checkpoint: {checkpoint_path}", "normal", verbosity)
        logger.load_checkpoint(checkpoint_path=checkpoint_path, model=model)
        collapsed_metrics = measure_baseline(model, tokenizer, device, verbosity=verbosity)
    else:
        vprint("No checkpoint provided. Running fresh HAEP attack to collapse model...", "normal", verbosity)
        from src.catalyst import HessianAwareCatalyst
        from src.attack_loop import MetabolicAttackLoop

        catalyst_gen = HessianAwareCatalyst(
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k_eigenvalues=config.hvp.num_null_directions,
        )
        attack_loop = MetabolicAttackLoop(
            model=model,
            catalyst_generator=catalyst_gen,
            device=device,
        )
        attack_loop.run_attack_cycle(
            num_iterations=config.attack.num_steps,
            target_rank_reduction=config.attack.target_rank_reduction,
            learning_rate=config.attack.learning_rate,
            catalyst_length=config.attack.catalyst_length,
            optimizer_type="adamw",
        )
        collapsed_metrics = measure_baseline(model, tokenizer, device, verbosity=verbosity)

    vprint(
        f"Collapsed state - Perplexity: {collapsed_metrics['perplexity']:.2f}, "
        f"Effective Rank: {collapsed_metrics['effective_rank']:.2f}",
        "normal",
        verbosity,
    )
    return model, tokenizer, original_baseline, collapsed_metrics


def load_clean_training_data(
    num_samples: int = 1000,
    max_length: int = 128,
    seed: int = 42,
    verbosity: str = "normal",
) -> list[dict]:
    """Load clean texts from Pile and tokenize for SFT."""
    vprint(f"Loading {num_samples} clean training samples from Pile...", "normal", verbosity)
    texts = load_random_pile_texts(num_samples=num_samples, seed=seed)
    if not texts:
        vprint("Warning: Pile load failed, using create_simple_test_set", "normal", verbosity)
        texts = create_simple_test_set(num_samples)
    return texts


def run_healing_loop(
    model,
    tokenizer,
    train_texts: list,
    device: str,
    healing_config: HealingConfig,
    collapsed_metrics: dict,
    original_baseline: dict,
    logger: ExperimentLogger,
    verbosity: str = "normal",
) -> list[dict]:
    """
    Run SFT healing loop on collapsed model.
    Returns healing history (list of dicts with step, effective_rank, perplexity, loss, etc.).
    """
    vprint(f"\nRunning healing loop: {healing_config.num_healing_steps} steps...", "normal", verbosity)

    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if healing_config.healing_optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=healing_config.healing_learning_rate)
    else:
        optimizer = torch.optim.SGD(trainable_params, lr=healing_config.healing_learning_rate)

    history = []
    interval = healing_config.measurement_interval
    max_len = healing_config.max_seq_length
    last_loss = torch.tensor(float("inf"), device=device)

    for step in range(healing_config.num_healing_steps):
        # Measure BEFORE training step at each interval (ensures step 0 is measured before any training)
        if step % interval == 0:
            model.eval()
            metrics = measure_baseline(model, tokenizer, device, num_samples=5, verbosity="quiet")
            recovery_pct = 0.0
            orig_r = original_baseline["effective_rank"]
            coll_r = collapsed_metrics["effective_rank"]
            curr_r = metrics["effective_rank"]
            if orig_r > coll_r and coll_r >= 0:
                recovery_pct = 100.0 * (curr_r - coll_r) / (orig_r - coll_r)
            recovery_pct = max(0.0, min(100.0, recovery_pct))

            loss_val = last_loss.item() if torch.isfinite(last_loss) else 1e10
            ppl_val = metrics["perplexity"]
            if not np.isfinite(ppl_val):
                ppl_val = 1e10
            entry = {
                "step": step,
                "effective_rank": metrics["effective_rank"],
                "perplexity": ppl_val,
                "spectral_gap": float(metrics.get("spectral_gap", 0)),
                "loss": loss_val,
                "recovery_pct": recovery_pct,
            }
            history.append(entry)
            logger.log_metrics(step, entry, prefix="healing")
            vprint(f"  Step {step}: rank={metrics['effective_rank']:.2f}, ppl={metrics['perplexity']:.2f}, recovery={recovery_pct:.1f}%", "verbose", verbosity)
            model.train()

        # Training step (padding=True to reduce NaN on collapsed models; shorter sequences)
        text_idx = step % len(train_texts)
        text = train_texts[text_idx]
        if isinstance(text, dict):
            text = text.get("text", str(text))
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding=True,
        ).to(device)
        input_ids = inputs["input_ids"]
        if input_ids.size(1) < 2:
            continue

        labels = input_ids[:, 1:].contiguous().clone()
        input_ids = input_ids[:, :-1].contiguous()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        last_loss = loss

        if torch.isfinite(loss):
            loss.backward()
            if healing_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, healing_config.gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    return history


def measure_recovery_metrics(
    final_metrics: dict,
    original_baseline: dict,
    collapsed_metrics: dict,
) -> dict:
    """Compute recovery percentage and final state summary."""
    orig_r = original_baseline["effective_rank"]
    coll_r = collapsed_metrics["effective_rank"]
    final_r = final_metrics["effective_rank"]
    rank_recovery_pct = 0.0
    if orig_r > coll_r and coll_r >= 0 and np.isfinite(final_r):
        rank_recovery_pct = 100.0 * (final_r - coll_r) / (orig_r - coll_r)
    rank_recovery_pct = max(0.0, min(100.0, rank_recovery_pct))

    orig_ppl = original_baseline["perplexity"]
    coll_ppl = collapsed_metrics["perplexity"]
    final_ppl = final_metrics["perplexity"]
    ppl_improvement_pct = 0.0
    if coll_ppl > 0 and np.isfinite(coll_ppl) and np.isfinite(final_ppl):
        ppl_improvement_pct = 100.0 * (coll_ppl - final_ppl) / coll_ppl

    return {
        "rank_recovery_pct": rank_recovery_pct,
        "ppl_improvement_pct": ppl_improvement_pct,
        "final_effective_rank": final_r,
        "final_perplexity": final_ppl,
        "original_rank": orig_r,
        "collapsed_rank": coll_r,
        "original_perplexity": orig_ppl,
        "collapsed_perplexity": coll_ppl,
    }


def run_healing_experiment(
    config: ExperimentConfig,
    healing_config: HealingConfig,
    args: argparse.Namespace,
    run_number: int = 1,
    checkpoint_path: str | None = None,
    checkpoint_label: str = "",
) -> dict:
    """Run a single healing experiment."""
    ckpt = checkpoint_path or getattr(args, "collapsed_checkpoint", None)
    torch.manual_seed(config.seed + run_number)
    np.random.seed(config.seed + run_number)

    device = args.device
    verbosity = args.verbosity

    experiment_name = f"pythia_{args.model}_healing"
    if checkpoint_label:
        experiment_name += f"_{checkpoint_label}"
    if getattr(args, "num_runs", 1) > 1 or getattr(args, "runs_per_checkpoint", 1) > 1:
        experiment_name += f"_run{run_number:02d}"

    output_dir = str(Path(args.output_dir) / "priority5_healing" / "raw_data")
    logger = ExperimentLogger(output_dir=output_dir, experiment_name=experiment_name)

    logger.log_config({
        "model": config.model.__dict__,
        "attack": config.attack.__dict__,
        "healing": {
            "num_healing_steps": healing_config.num_healing_steps,
            "healing_learning_rate": healing_config.healing_learning_rate,
            "healing_optimizer": healing_config.healing_optimizer,
            "measurement_interval": healing_config.measurement_interval,
            "num_training_samples": healing_config.num_training_samples,
        },
    })

    model, tokenizer, original_baseline, collapsed_metrics = load_collapsed_model(
        checkpoint_path=ckpt,
        config=config,
        device=device,
        logger=logger,
        verbosity=verbosity,
        no_gradient_checkpointing=args.no_gradient_checkpointing,
    )

    train_texts = load_clean_training_data(
        num_samples=healing_config.num_training_samples,
        max_length=healing_config.max_seq_length,
        seed=config.seed + run_number,
        verbosity=verbosity,
    )

    healing_history = run_healing_loop(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        device=device,
        healing_config=healing_config,
        collapsed_metrics=collapsed_metrics,
        original_baseline=original_baseline,
        logger=logger,
        verbosity=verbosity,
    )

    final_metrics = measure_baseline(model, tokenizer, device, verbosity=verbosity)
    recovery = measure_recovery_metrics(final_metrics, original_baseline, collapsed_metrics)

    log_path = logger.get_log_path()
    with open(log_path / "healing_summary.json", "w") as f:
        serializable = {}
        for k, v in recovery.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (float, np.floating)) and not np.isfinite(v):
                serializable[k] = None  # JSON does not support inf/nan
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)

    plot_recovery_trajectory(
        healing_history=healing_history,
        original_rank=original_baseline["effective_rank"],
        collapsed_rank=collapsed_metrics["effective_rank"],
        save_path=str(log_path / "recovery_trajectory.png"),
    )
    plot_perplexity_recovery(
        healing_history=healing_history,
        original_ppl=original_baseline["perplexity"],
        collapsed_ppl=collapsed_metrics["perplexity"],
        save_path=str(log_path / "perplexity_recovery.png"),
    )
    orig_sv = original_baseline.get("singular_values", np.array([]))
    coll_sv = collapsed_metrics.get("singular_values", np.array([]))
    healed_sv = final_metrics.get("singular_values", np.array([]))
    if len(orig_sv) > 0 and len(coll_sv) > 0 and len(healed_sv) > 0:
        plot_spectral_decay_three_state(
            original_sv=orig_sv,
            collapsed_sv=coll_sv,
            healed_sv=healed_sv,
            save_path=str(log_path / "spectral_decay_three_state.png"),
        )

    vprint("\n" + "=" * 50, "normal", verbosity)
    vprint("HEALING EXPERIMENT SUMMARY", "normal", verbosity)
    vprint("=" * 50, "normal", verbosity)
    vprint(f"Rank recovery: {recovery['rank_recovery_pct']:.1f}%", "normal", verbosity)
    vprint(f"Original rank: {original_baseline['effective_rank']:.2f}", "normal", verbosity)
    vprint(f"Collapsed rank: {collapsed_metrics['effective_rank']:.2f}", "normal", verbosity)
    vprint(f"Final rank: {final_metrics['effective_rank']:.2f}", "normal", verbosity)
    vprint(f"Results saved to: {log_path}", "quiet", verbosity)

    return {
        "run_number": run_number,
        "checkpoint_path": str(ckpt) if ckpt else None,
        "checkpoint_label": checkpoint_label,
        "rank_recovery_pct": recovery["rank_recovery_pct"],
        "ppl_improvement_pct": recovery["ppl_improvement_pct"],
        "final_rank": final_metrics["effective_rank"],
        "final_perplexity": final_metrics["perplexity"],
        "log_path": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run healing/reversibility experiment")
    parser.add_argument("--model", type=str, choices=["70m", "160m", "410m", "1b", "1.4b", "2.8b"], default="1b")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./experiments/results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--collapsed-checkpoint", type=str, default=None, help="Path to single collapsed checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to scan for checkpoint_post_treatment.pt (overrides --collapsed-checkpoint)")
    parser.add_argument("--runs-per-checkpoint", type=int, default=1, choices=[1, 2, 3],
                        help="Runs per checkpoint when using --checkpoint-dir (default: 1)")
    parser.add_argument("--healing-steps", type=int, default=500, help="Number of SFT steps for healing")
    parser.add_argument("--healing-lr", type=float, default=1e-4, help="Learning rate for healing SFT")
    parser.add_argument("--healing-optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gradient-clip", type=float, default=0.0,
                        help="Gradient clipping threshold (0.0 = no clipping, 1.0 = aggressive)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs when using single checkpoint")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (may increase NaN on collapsed models)")
    parser.add_argument("--verbosity", choices=["quiet", "normal", "verbose"], default="normal",
                        help="quiet: minimal output; normal: standard; verbose: per-step progress")
    parser.add_argument("-q", "--quiet", action="store_const", dest="verbosity", const="quiet",
                        help="Shorthand for --verbosity quiet")

    args = parser.parse_args()
    args.no_gradient_checkpointing = not args.gradient_checkpointing

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        configs = {
            "70m": get_pythia_70m_config,
            "160m": get_pythia_160m_config,
            "410m": get_pythia_410m_config,
            "1b": get_pythia_1b_config,
            "1.4b": get_pythia_1_4b_config,
            "2.8b": get_pythia_2_8b_config,
        }
        config = configs[args.model]()

    config.device = args.device

    healing_config = HealingConfig(
        num_healing_steps=args.healing_steps,
        healing_learning_rate=args.healing_lr,
        healing_optimizer=args.healing_optimizer,
        gradient_clip=args.gradient_clip,
        measurement_interval=10,
        num_training_samples=1000,
        max_seq_length=128,
    )

    all_results = []

    if args.checkpoint_dir:
        # Multi-checkpoint mode: discover checkpoints and run each
        checkpoints = discover_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            vprint(f"No checkpoint_post_treatment.pt found under {args.checkpoint_dir}", "normal", args.verbosity)
            return
        vprint(f"Found {len(checkpoints)} checkpoint(s) under {args.checkpoint_dir}", "normal", args.verbosity)
        total_runs = len(checkpoints) * args.runs_per_checkpoint
        run_idx = 0
        for ckpt_path in checkpoints:
            # Derive label from path, e.g. pythia_1b_run01_fp16
            parent = ckpt_path.parent.parent  # .../checkpoints/ -> .../pythia_1b_run01_fp16/
            checkpoint_label = parent.name if parent.name else ckpt_path.stem
            for run_num in range(1, args.runs_per_checkpoint + 1):
                run_idx += 1
                vprint(f"\n{'#'*50}\nCHECKPOINT {run_idx}/{total_runs}: {ckpt_path.name} (run {run_num}/{args.runs_per_checkpoint})\n{'#'*50}", "normal", args.verbosity)
                try:
                    result = run_healing_experiment(
                        config, healing_config, args, run_num,
                        checkpoint_path=str(ckpt_path),
                        checkpoint_label=checkpoint_label,
                    )
                    all_results.append(result)
                except Exception as e:
                    vprint(f"ERROR: {e}", "normal", args.verbosity)
                    if args.verbosity == "verbose":
                        import traceback
                        traceback.print_exc()
    else:
        # Single checkpoint mode
        if not args.collapsed_checkpoint:
            vprint("Provide --collapsed-checkpoint or --checkpoint-dir", "normal", args.verbosity)
            return
        for run_num in range(1, args.num_runs + 1):
            vprint(f"\n{'#'*50}\nRUN {run_num}/{args.num_runs}\n{'#'*50}", "normal", args.verbosity)
            try:
                result = run_healing_experiment(config, healing_config, args, run_num)
                all_results.append(result)
            except Exception as e:
                vprint(f"ERROR in run {run_num}: {e}", "normal", args.verbosity)
                if args.verbosity == "verbose":
                    import traceback
                    traceback.print_exc()

    if all_results and len(all_results) > 1:
        import statistics
        recoveries = [r["rank_recovery_pct"] for r in all_results]
        std_str = f" std={statistics.stdev(recoveries):.1f}%" if len(recoveries) >= 2 else ""
        vprint(f"\nAggregate ({len(all_results)} runs): rank_recovery mean={statistics.mean(recoveries):.1f}%{std_str}", "normal", args.verbosity)


if __name__ == "__main__":
    main()
