"""
Metabolic Attack Loop

Simulates the "metabolic cycle" where repeated exposure to catalysts
induces progressive degradation through spectral collapse.
"""

import torch
from typing import Dict, List, Optional, Union
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from transformers.optimization import Adafactor
    ADAFACTOR_AVAILABLE = True
except ImportError:
    ADAFACTOR_AVAILABLE = False

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
        learning_rate: float = 1e-4,
        catalyst_length: int = 64,
        optimizer_type: str = 'adamw'
    ) -> Dict:
        """
        Run a single attack cycle.
        
        Args:
            num_iterations: Number of forward passes with catalyst
            catalyst_tokens: Pre-generated catalyst tokens (if None, generate new)
            target_rank_reduction: Target reduction in effective rank
            learning_rate: Learning rate for parameter updates
            catalyst_length: Length of catalyst sequence in tokens
            optimizer_type: Type of optimizer to use ('adamw' or 'sgd')
            
        Returns:
            attack_results: Dictionary with attack metrics
        """
        # Generate catalyst if not provided
        if catalyst_tokens is None:
            # Clear memory before catalyst generation
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            catalyst_tokens = self.catalyst_generator.generate_catalyst(
                num_steps=50,
                learning_rate=1e-2,
                catalyst_length=catalyst_length
            )
            # Clear memory after catalyst generation
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Initialize optimizer based on type
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if trainable_params:
            if optimizer_type == 'sgd':
                # Priority 3: SGD without momentum or adaptive learning rate
                optimizer = torch.optim.SGD(
                    trainable_params,
                    lr=learning_rate,
                    momentum=0.0,  # No momentum (per Priority 3 spec)
                    dampening=0.0,
                    weight_decay=0.0,
                    nesterov=False
                )
                print(f"Using SGD optimizer (no momentum, no adaptive LR) for {len(trainable_params)} trainable parameters")
            elif optimizer_type == 'adamw':
                # Existing AdamW selection logic (Adafactor -> AdamW8bit -> AdamW)
                # Option 2 (PI directive): Use Adafactor for zero optimizer state memory overhead
                # This eliminates the ~2.8 GB optimizer state memory for 1.4B models
                if ADAFACTOR_AVAILABLE:
                    # Adafactor: Zero optimizer state overhead (stores row/column sums instead of full states)
                    # Trade-off: Slightly slower convergence, but acceptable for 100-step attack
                    optimizer = Adafactor(
                        trainable_params,
                        lr=learning_rate,
                        relative_step=False,  # Use fixed learning rate
                        scale_parameter=False,  # Disable parameter scaling
                        warmup_init=False
                    )
                    print(f"Using Adafactor optimizer (zero optimizer state overhead) for {len(trainable_params)} trainable parameters")
                elif BITSANDBYTES_AVAILABLE:
                    # Fallback: 8-bit AdamW optimizer (saves ~8.4 GB VRAM vs FP32 Adam)
                    optimizer = bnb.optim.AdamW8bit(
                        trainable_params,
                        lr=learning_rate,
                        betas=(0.9, 0.999),
                        eps=1e-8
                    )
                    print(f"Using AdamW8bit optimizer (8-bit optimizer states) for {len(trainable_params)} trainable parameters")
                else:
                    # Final fallback: Standard AdamW if neither Adafactor nor bitsandbytes available
                    optimizer = torch.optim.AdamW(
                        trainable_params,
                        lr=learning_rate,
                        betas=(0.9, 0.999),
                        eps=1e-8
                    )
                    print(f"Using standard AdamW optimizer (FP32 optimizer states) for {len(trainable_params)} trainable parameters")
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}. Must be 'adamw' or 'sgd'")
        else:
            optimizer = None
            print("Warning: No trainable parameters found, optimizer not initialized")
        
        # Measure initial rank
        initial_rank = self._measure_current_rank(catalyst_tokens)
        
        # Simulate repeated exposure
        for iteration in tqdm(range(num_iterations), desc="Attack cycle"):
            # Expose model to catalyst
            self._expose_to_catalyst(catalyst_tokens, optimizer)
            
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
        
        # Clean up optimizer
        if optimizer is not None:
            del optimizer
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Expose model to catalyst tokens.
        
        This performs a forward and backward pass, updating model parameters
        using the provided optimizer (AdamW8bit for memory efficiency).
        
        Args:
            catalyst_tokens: Token IDs for catalyst
            optimizer: Optimizer instance (AdamW8bit or AdamW). If None, falls back to manual SGD.
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
        
        # Update parameters using optimizer (memory-efficient 8-bit AdamW)
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        else:
            # Fallback to manual SGD if optimizer not provided (for backward compatibility)
            # This should not happen in normal operation after refactoring
            learning_rate = 1e-4  # Default fallback LR
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.data -= learning_rate * param.grad
                        param.grad.zero_()
        
        # Clear output tensors and free memory
        del outputs, loss
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
