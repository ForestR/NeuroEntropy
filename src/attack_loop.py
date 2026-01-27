"""
Metabolic Attack Loop

Simulates the "metabolic cycle" where repeated exposure to catalysts
induces progressive degradation through spectral collapse.
"""

import torch
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from .catalyst import HessianAwareCatalyst
from .diagnosis import compute_effective_rank, compute_spectral_gap, ActivationHook


class MetabolicAttackLoop:
    """
    Implements the iterative attack process that simulates metabolic degradation.
    
    The attack works by:
    1. Generating catalysts that exploit Hessian structure
    2. Exposing model to catalysts repeatedly
    3. Monitoring spectral collapse (effective rank reduction)
    4. Continuing until target degradation is reached
    """
    
    def __init__(
        self,
        model,
        catalyst_generator: HessianAwareCatalyst,
        device: str = "cuda",
        use_lora: bool = False
    ):
        """
        Initialize the attack loop.
        
        Args:
            model: Target model to attack
            catalyst_generator: Generator for attack catalysts
            device: Computing device
            use_lora: Whether to use LoRA for attacks
        """
        self.model = model
        self.catalyst_generator = catalyst_generator
        self.device = device
        self.use_lora = use_lora
        self.history: List[Dict] = []
        
        # Setup activation hook for rank measurement
        self.activation_hook = ActivationHook()
        self._setup_activation_hook()
    
    def _setup_activation_hook(self):
        """Setup activation hook for measuring effective rank."""
        try:
            self.activation_hook.register(self.model)
        except Exception as e:
            print(f"Warning: Could not setup activation hook: {e}")
    
    def run_attack_cycle(
        self,
        num_iterations: int = 100,
        catalyst_tokens: Optional[torch.Tensor] = None,
        target_rank_reduction: float = 0.5,
        learning_rate: float = 1e-4
    ) -> Dict:
        """
        Run a single attack cycle.
        
        Args:
            num_iterations: Number of forward passes with catalyst
            catalyst_tokens: Pre-generated catalyst tokens (if None, generate new)
            target_rank_reduction: Target reduction in effective rank
            learning_rate: Learning rate for parameter updates
            
        Returns:
            attack_results: Dictionary with attack metrics
        """
        # Generate catalyst if not provided
        if catalyst_tokens is None:
            catalyst_tokens = self.catalyst_generator.generate_catalyst(
                num_steps=50,
                learning_rate=1e-2,
                catalyst_length=128
            )
        
        # Measure initial rank
        initial_rank = self._measure_current_rank(catalyst_tokens)
        
        # Simulate repeated exposure
        for iteration in tqdm(range(num_iterations), desc="Attack cycle"):
            # Expose model to catalyst
            self._expose_to_catalyst(catalyst_tokens, learning_rate)
            
            # Periodic diagnosis
            if iteration % 10 == 0 or iteration == num_iterations - 1:
                current_rank = self._measure_current_rank(catalyst_tokens)
                rank_reduction = 1.0 - (current_rank / initial_rank) if initial_rank > 0 else 0.0
                
                self.history.append({
                    'iteration': iteration,
                    'effective_rank': current_rank,
                    'rank_reduction': rank_reduction
                })
                
                if rank_reduction >= target_rank_reduction:
                    break
        
        final_rank = self._measure_current_rank(catalyst_tokens)
        
        return {
            'initial_rank': initial_rank,
            'final_rank': final_rank,
            'rank_reduction': 1.0 - (final_rank / initial_rank) if initial_rank > 0 else 0.0,
            'iterations': len(self.history),
            'catalyst_tokens': catalyst_tokens
        }
    
    def _expose_to_catalyst(
        self,
        catalyst_tokens: torch.Tensor,
        learning_rate: float = 1e-4
    ):
        """
        Expose model to catalyst tokens.
        
        This performs a forward and backward pass, updating model parameters
        (or LoRA parameters if use_lora=True).
        
        Args:
            catalyst_tokens: Token IDs for catalyst
            learning_rate: Learning rate for parameter updates
        """
        self.model.train()
        
        # Move tokens to device if needed
        if catalyst_tokens.device != self.device:
            catalyst_tokens = catalyst_tokens.to(self.device)
        
        # Prepare labels (shifted by 1 for language modeling)
        input_ids = catalyst_tokens[:, :-1]
        labels = catalyst_tokens[:, 1:]
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
        
        self.model.eval()
    
    def _measure_current_rank(
        self,
        input_tokens: Optional[torch.Tensor] = None
    ) -> float:
        """
        Measure current effective rank of model activations.
        
        Args:
            input_tokens: Tokens to use for measurement (if None, use random)
            
        Returns:
            effective_rank: Current effective rank value
        """
        self.model.eval()
        self.activation_hook.clear()
        
        # Re-register hook after clearing
        try:
            self.activation_hook.register(self.model)
        except Exception as e:
            print(f"Warning: Could not re-register activation hook: {e}")
            return 0.0
        
        # Use provided tokens or generate random ones
        if input_tokens is None:
            vocab_size = len(self.catalyst_generator.tokenizer)
            input_tokens = torch.randint(
                0, vocab_size,
                (1, 128),
                device=self.device
            )
        else:
            input_tokens = input_tokens.to(self.device)
        
        # Forward pass to collect activations
        with torch.no_grad():
            try:
                _ = self.model(input_ids=input_tokens)
            except Exception as e:
                print(f"Warning: Error during rank measurement: {e}")
                return 0.0
        
        # Get collected activations
        activations = self.activation_hook.get_activations()
        
        if activations is None or activations.numel() == 0:
            return 0.0
        
        # Compute effective rank
        effective_rank = compute_effective_rank(activations)
        return effective_rank
    
    def get_history(self) -> List[Dict]:
        """Get attack history."""
        return self.history
    
    def clear_history(self):
        """Clear attack history."""
        self.history = []
