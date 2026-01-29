"""
Experiment logging utilities.

Provides logging functionality for tracking experiments, metrics, and checkpoints.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import torch
import numpy as np


class ExperimentLogger:
    """
    Logger for tracking experiment metrics and checkpoints.
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
        resume: bool = False
    ):
        """
        Initialize experiment logger.
        
        Args:
            output_dir: Directory to save logs and checkpoints
            experiment_name: Name of experiment (if None, auto-generate)
            resume: Whether to resume from existing log
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_dir = self.output_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.config_file = self.log_dir / "config.json"
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize metrics log
        if not resume:
            self.metrics_file.unlink(missing_ok=True)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Convert non-serializable types
        serializable_config = self._make_serializable(config)
        
        with open(self.config_file, 'w') as f:
            json.dump(serializable_config, f, indent=2)
    
    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        Log metrics at a given step.
        
        Args:
            step: Step number
            metrics: Dictionary of metric names and values
            prefix: Optional prefix for metric names
        """
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **{f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
        }
        
        # Convert non-serializable types (numpy arrays, torch tensors, etc.)
        serializable_log_entry = self._make_serializable(log_entry)
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(serializable_log_entry) + '\n')
    
    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        additional_state: Optional[Dict[str, Any]] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            step: Step number
            model: Model to save
            optimizer: Optimizer state (optional)
            additional_state: Additional state to save (optional)
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if additional_state is not None:
            checkpoint.update(additional_state)
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (if None, load latest)
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            checkpoint: Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if model is not None:
            state_dict = checkpoint['model_state_dict']
            
            # Filter out quantization-related keys (bitsandbytes metadata)
            # These keys are not part of the standard PyTorch model structure
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not any(quant_key in k for quant_key in ['.absmax', '.quant_map', '.quant_state'])
            }
            
            # Load with strict=False to handle any remaining mismatches gracefully
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            # Log warnings if there are missing or unexpected keys (but quantization keys are expected)
            if missing_keys:
                # Filter out quantization-related missing keys from warnings
                non_quant_missing = [
                    k for k in missing_keys 
                    if not any(quant_key in k for quant_key in ['.absmax', '.quant_map', '.quant_state'])
                ]
                if non_quant_missing:
                    print(f"Warning: Some model parameters were not found in checkpoint: {non_quant_missing[:5]}{'...' if len(non_quant_missing) > 5 else ''}")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return str(obj)
    
    def get_log_path(self) -> Path:
        """Get path to log directory."""
        return self.log_dir
