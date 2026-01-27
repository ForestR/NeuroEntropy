"""
LoRA Poisoning Protocol

Implements Low-Rank Adaptation (LoRA) based metabolic attack.
Forces high-rank updates by aligning LoRA matrices with noise dimensions.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import PreTrainedModel


class LoRAPoisoning:
    """
    LoRA-based poisoning attack that forces high-rank updates.
    
    The key insight: By aligning LoRA matrices (A, B) with noise dimensions
    from the null space, we can force the model to expand its energy profile
    even under constrained low-rank updates.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        rank: int = 16,
        alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        device: str = "cuda"
    ):
        """
        Initialize LoRA poisoning module.
        
        Args:
            model: Base model to attack
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            target_modules: List of module names to apply LoRA (if None, auto-detect)
            device: Computing device
        """
        self.base_model = model
        self.device = device
        self.rank = rank
        self.alpha = alpha
        
        # Auto-detect target modules if not specified
        if target_modules is None:
            target_modules = self._auto_detect_modules(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(model, lora_config)
        self.model.to(device)
        
        # Track LoRA parameters
        self.lora_params = self._get_lora_parameters()
    
    def _auto_detect_modules(self, model: PreTrainedModel) -> List[str]:
        """
        Auto-detect target modules for LoRA.
        
        Common patterns:
        - GPT models: "q_proj", "v_proj", "k_proj", "out_proj"
        - LLaMA models: "q_proj", "v_proj", "k_proj", "o_proj"
        """
        module_names = []
        
        # Try to find attention modules
        for name, module in model.named_modules():
            if any(key in name.lower() for key in ['q_proj', 'v_proj', 'k_proj', 'out_proj', 'o_proj']):
                module_names.append(name)
        
        # If nothing found, use a common pattern
        if not module_names:
            module_names = ["q_proj", "v_proj"]
        
        return module_names[:4]  # Limit to first 4 modules
    
    def _get_lora_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Get LoRA parameters (A and B matrices)."""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                lora_params[name] = param
        return lora_params
    
    def compute_lora_rank(self) -> float:
        """
        Compute effective rank of LoRA matrices.
        
        Returns average rank across all LoRA modules.
        """
        ranks = []
        
        for name, param in self.lora_params.items():
            if 'lora_A' in name or 'lora_B' in name:
                # Reshape to matrix
                if param.dim() > 1:
                    matrix = param.view(param.size(0), -1)
                else:
                    continue
                
                # Compute effective rank using SVD
                try:
                    U, s, Vt = torch.linalg.svd(matrix, full_matrices=False)
                    # Effective rank as number of significant singular values
                    threshold = s[0] * 1e-3
                    rank = (s > threshold).sum().item()
                    ranks.append(rank)
                except Exception:
                    continue
        
        return sum(ranks) / len(ranks) if ranks else 0.0
    
    def align_with_null_space(
        self,
        null_directions: List[torch.Tensor],
        strength: float = 1.0
    ):
        """
        Align LoRA matrices with null space directions.
        
        This forces the LoRA update to expand into high-energy noise dimensions.
        
        Args:
            null_directions: List of null space direction vectors
            strength: Alignment strength (0-1)
        """
        self.model.train()
        
        for name, param in self.lora_params.items():
            if param.requires_grad:
                # Get current parameter values
                current = param.data.clone()
                
                # Project onto null space (simplified - in practice would
                # need proper mapping between parameter space and null space)
                # For now, add noise aligned with null directions
                noise = torch.zeros_like(current)
                
                for null_dir in null_directions[:3]:  # Use first 3 null directions
                    # Simplified projection (would need proper dimension matching)
                    if current.numel() <= null_dir.numel():
                        null_proj = null_dir[:current.numel()].view_as(current)
                        noise += strength * null_proj
                
                # Update parameter
                param.data = current + noise
    
    def poison_step(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        null_directions: Optional[List[torch.Tensor]] = None,
        learning_rate: float = 1e-4
    ) -> Dict[str, float]:
        """
        Perform one poisoning step.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels (if None, use shifted input_ids)
            null_directions: Null space directions for alignment
            learning_rate: Learning rate for update
            
        Returns:
            metrics: Dictionary with step metrics
        """
        self.model.train()
        
        # Prepare labels
        if labels is None:
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Align gradients with null space if provided
        if null_directions:
            self._align_gradients_with_null_space(null_directions)
        
        # Update parameters (using standard optimizer)
        # In practice, would use custom optimizer that amplifies noise
        with torch.no_grad():
            for name, param in self.lora_params.items():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
        
        # Compute metrics
        lora_rank = self.compute_lora_rank()
        
        return {
            'loss': loss.item(),
            'lora_rank': lora_rank,
        }
    
    def _align_gradients_with_null_space(
        self,
        null_directions: List[torch.Tensor]
    ):
        """
        Align parameter gradients with null space directions.
        
        This amplifies noise in null directions, exploiting Adam's
        second-moment normalization.
        """
        for name, param in self.lora_params.items():
            if param.grad is not None:
                grad = param.grad.data
                
                # Project gradient onto null space
                # Simplified: add component aligned with null directions
                for null_dir in null_directions[:2]:  # Use first 2 null directions
                    if grad.numel() <= null_dir.numel():
                        null_proj = null_dir[:grad.numel()].view_as(grad)
                        # Amplify component in null direction
                        grad += 0.1 * null_proj * torch.norm(grad)
    
    def get_model(self) -> PeftModel:
        """Get the LoRA-wrapped model."""
        return self.model
    
    def save_lora(self, path: str):
        """Save LoRA adapters."""
        self.model.save_pretrained(path)
    
    def load_lora(self, path: str):
        """Load LoRA adapters."""
        self.model = PeftModel.from_pretrained(self.base_model, path)
        self.model.to(self.device)
        self.lora_params = self._get_lora_parameters()
