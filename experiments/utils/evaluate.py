"""
Evaluation utilities for model performance metrics.

Provides functions for evaluating model degradation, including perplexity
and simplified zero-shot evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512
) -> float:
    """
    Compute perplexity on a list of texts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of input texts
        device: Computing device
        max_length: Maximum sequence length
        
    Returns:
        perplexity: Average perplexity
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(device)
            
            input_ids = inputs['input_ids']
            if input_ids.size(1) < 2:
                continue
            
            # Prepare labels
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Check for numerical instability
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping sample")
                continue
            
            # Clip loss to prevent overflow in exp()
            # exp(50) ≈ 5e21, still manageable for float64
            loss_clipped = torch.clamp(loss, max=50.0)
            
            # Accumulate
            num_tokens = labels.numel()
            total_loss += loss_clipped.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    
    # Final check before exp()
    if not np.isfinite(avg_loss):
        print(f"Warning: Non-finite average loss: {avg_loss}, returning inf")
        return float('inf')
    
    # Clip avg_loss before exp to prevent overflow
    avg_loss_clipped = min(avg_loss, 50.0)
    perplexity = np.exp(avg_loss_clipped)
    
    return perplexity


def evaluate_zero_shot_simple(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    completions: List[List[str]],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Simple zero-shot evaluation: measure likelihood of correct completions.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        completions: List of completion options (each is a list of strings)
        device: Computing device
        
    Returns:
        metrics: Dictionary with accuracy and average log-likelihood
    """
    model.eval()
    correct = 0
    total = 0
    all_log_probs = []
    
    with torch.no_grad():
        for prompt, completion_options in zip(prompts, completions):
            if not completion_options:
                continue
            
            best_log_prob = float('-inf')
            best_idx = -1
            
            for idx, completion in enumerate(completion_options):
                # Combine prompt and completion
                full_text = prompt + " " + completion
                
                # Tokenize
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)
                
                input_ids = inputs['input_ids']
                if input_ids.size(1) < 2:
                    continue
                
                # Get log probabilities
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                
                # Compute log probability of completion tokens
                prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
                completion_start = prompt_tokens.size(1) - 1
                
                if completion_start >= input_ids.size(1):
                    continue
                
                completion_tokens = input_ids[0, completion_start:]
                completion_logits = logits[0, completion_start-1:-1]
                
                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
                token_log_probs = log_probs.gather(1, completion_tokens.unsqueeze(1)).squeeze(1)
                avg_log_prob = token_log_probs.mean().item()
                
                all_log_probs.append(avg_log_prob)
                
                if avg_log_prob > best_log_prob:
                    best_log_prob = avg_log_prob
                    best_idx = idx
            
            if best_idx == 0:  # Assuming first option is correct
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    avg_log_prob = np.mean(all_log_probs) if all_log_probs else 0.0
    
    return {
        'accuracy': accuracy,
        'avg_log_probability': avg_log_prob,
        'num_examples': total
    }


def compare_before_after(
    model_before: PreTrainedModel,
    model_after: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_texts: List[str],
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Compare model performance before and after attack.
    
    Args:
        model_before: Model before attack
        model_after: Model after attack
        tokenizer: Tokenizer
        test_texts: List of test texts
        device: Computing device
        
    Returns:
        comparison: Dictionary with before/after metrics
    """
    print("Evaluating model before attack...")
    perplexity_before = compute_perplexity(model_before, tokenizer, test_texts, device)
    
    print("Evaluating model after attack...")
    perplexity_after = compute_perplexity(model_after, tokenizer, test_texts, device)
    
    perplexity_increase = perplexity_after / perplexity_before if perplexity_before > 0 else float('inf')
    
    return {
        'before': {
            'perplexity': perplexity_before
        },
        'after': {
            'perplexity': perplexity_after
        },
        'degradation': {
            'perplexity_increase': perplexity_increase,
            'perplexity_increase_pct': (perplexity_increase - 1) * 100
        }
    }


def create_simple_test_set(num_examples: int = 100) -> List[str]:
    """
    Create a simple test set for evaluation.
    
    Args:
        num_examples: Number of examples to generate
        
    Returns:
        texts: List of test texts
    """
    # Simple test texts (in practice, would use a real dataset)
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "To be or not to be, that is the question.",
        "The answer to the ultimate question of life, the universe, and everything is 42.",
    ]
    
    # Repeat and vary
    texts = []
    for i in range(num_examples):
        base_text = base_texts[i % len(base_texts)]
        texts.append(base_text)
    
    return texts
