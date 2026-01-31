"""
Configuration management for NeuroEntropy experiments.

Centralized configuration for models, LoRA, attacks, and HVP parameters.
"""

from dataclasses import dataclass
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str  # e.g., "pythia-70m", "pythia-160m"
    hf_model_id: str  # HuggingFace model ID
    max_seq_length: int = 512
    use_quantization: bool = False
    quantization_bits: int = 4  # 4-bit or 8-bit
    use_gradient_checkpointing: bool = False
    use_fft: bool = False  # Full Fine-Tuning mode (no LoRA)


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None  # If None, auto-detect
    bias: str = "none"  # "none", "all", "lora_only"


@dataclass
class AttackConfig:
    """Attack configuration."""
    num_steps: int = 100  # Number of gradient steps
    learning_rate: float = 1e-4
    batch_size: int = 1
    target_rank_reduction: float = 0.5  # Target reduction in effective rank
    catalyst_length: int = 64  # Length of catalyst prompt in tokens (reduced from 128 for memory efficiency)


@dataclass
class HVPConfig:
    """Hessian-Vector Product configuration."""
    num_power_iterations: int = 50
    null_space_threshold: float = 1e-6  # Threshold for detecting null space
    num_null_directions: int = 10  # Number of null directions to find
    hvp_batch_size: int = 1  # Batch size for HVP computation


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    lora: LoRAConfig
    attack: AttackConfig
    hvp: HVPConfig
    output_dir: str = "./experiments/results"
    seed: int = 42
    device: str = "cuda"
    save_checkpoints: bool = True
    checkpoint_interval: int = 10  # Save checkpoint every N steps
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_dict = config_dict['model']
        model_dict.setdefault('use_gradient_checkpointing', False)
        model_dict.setdefault('use_fft', False)
        return cls(
            model=ModelConfig(**model_dict),
            lora=LoRAConfig(**config_dict.get('lora', {})),
            attack=AttackConfig(**config_dict.get('attack', {})),
            hvp=HVPConfig(**config_dict.get('hvp', {})),
            output_dir=config_dict.get('output_dir', './experiments/results'),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
            save_checkpoints=config_dict.get('save_checkpoints', True),
            checkpoint_interval=config_dict.get('checkpoint_interval', 10),
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'model': {
                'name': self.model.name,
                'hf_model_id': self.model.hf_model_id,
                'max_seq_length': self.model.max_seq_length,
                'use_quantization': self.model.use_quantization,
                'quantization_bits': self.model.quantization_bits,
                'use_gradient_checkpointing': self.model.use_gradient_checkpointing,
                'use_fft': self.model.use_fft,
            },
            'lora': {
                'rank': self.lora.rank,
                'alpha': self.lora.alpha,
                'dropout': self.lora.dropout,
                'target_modules': self.lora.target_modules,
                'bias': self.lora.bias,
            },
            'attack': {
                'num_steps': self.attack.num_steps,
                'learning_rate': self.attack.learning_rate,
                'batch_size': self.attack.batch_size,
                'target_rank_reduction': self.attack.target_rank_reduction,
                'catalyst_length': self.attack.catalyst_length,
            },
            'hvp': {
                'num_power_iterations': self.hvp.num_power_iterations,
                'null_space_threshold': self.hvp.null_space_threshold,
                'num_null_directions': self.hvp.num_null_directions,
                'hvp_batch_size': self.hvp.hvp_batch_size,
            },
            'output_dir': self.output_dir,
            'seed': self.seed,
            'device': self.device,
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_interval': self.checkpoint_interval,
        }
        
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configurations for common experiments
def get_pythia_70m_config() -> ExperimentConfig:
    """Get default config for Pythia-70M."""
    return ExperimentConfig(
        model=ModelConfig(
            name="pythia-70m",
            hf_model_id="EleutherAI/pythia-70m",
            max_seq_length=512,
            use_quantization=False,
        ),
        lora=LoRAConfig(rank=8, alpha=16),
        attack=AttackConfig(num_steps=100, learning_rate=1e-4),
        hvp=HVPConfig(num_power_iterations=50, null_space_threshold=1e-6),
    )


def get_pythia_160m_config() -> ExperimentConfig:
    """Get default config for Pythia-160M."""
    return ExperimentConfig(
        model=ModelConfig(
            name="pythia-160m",
            hf_model_id="EleutherAI/pythia-160m",
            max_seq_length=512,
            use_quantization=False,
        ),
        lora=LoRAConfig(rank=16, alpha=32),
        attack=AttackConfig(num_steps=100, learning_rate=1e-4),
        hvp=HVPConfig(num_power_iterations=50, null_space_threshold=1e-6),
    )


def get_pythia_410m_config() -> ExperimentConfig:
    """Get default config for Pythia-410M."""
    return ExperimentConfig(
        model=ModelConfig(
            name="pythia-410m",
            hf_model_id="EleutherAI/pythia-410m",
            max_seq_length=512,
            use_quantization=True,  # Use quantization for larger model
            quantization_bits=4,
        ),
        lora=LoRAConfig(rank=16, alpha=32),
        attack=AttackConfig(num_steps=100, learning_rate=1e-4),
        hvp=HVPConfig(num_power_iterations=50, null_space_threshold=1e-6),
    )


def get_pythia_1b_config() -> ExperimentConfig:
    """Get default config for Pythia-1B (FFT mode per PI directive)."""
    return ExperimentConfig(
        model=ModelConfig(
            name="pythia-1b",
            hf_model_id="EleutherAI/pythia-1b",
            max_seq_length=512,
            use_quantization=False,  # FP16 per Priority 1
            use_gradient_checkpointing=True,
            use_fft=True,
        ),
        lora=LoRAConfig(rank=16, alpha=32),
        attack=AttackConfig(
            num_steps=100,
            learning_rate=2e-4,  # Increased from 1e-4 for larger models (compensate for relative perturbation decrease)
            catalyst_length=64  # Reduced from 128 for memory efficiency
        ),
        hvp=HVPConfig(num_power_iterations=50, null_space_threshold=1e-6),
    )


def get_pythia_1_4b_config() -> ExperimentConfig:
    """Get default config for Pythia-1.4B (FFT mode per PI directive)."""
    return ExperimentConfig(
        model=ModelConfig(
            name="pythia-1.4b",
            hf_model_id="EleutherAI/pythia-1.4b",
            max_seq_length=512,
            use_quantization=False,  # FP16 per Priority 1
            use_gradient_checkpointing=True,
            use_fft=True,
        ),
        lora=LoRAConfig(rank=16, alpha=32),
        attack=AttackConfig(
            num_steps=100,
            learning_rate=2e-4,  # Increased from 1e-4 for larger models (compensate for relative perturbation decrease)
            catalyst_length=32  # Aggressive reduction for 1.4B memory constraints (PI directive: Option 1)
        ),
        hvp=HVPConfig(num_power_iterations=50, null_space_threshold=1e-6),
    )


def get_pythia_2_8b_config() -> ExperimentConfig:
    """Get default config for Pythia-2.8B."""
    return ExperimentConfig(
        model=ModelConfig(
            name="pythia-2.8b",
            hf_model_id="EleutherAI/pythia-2.8b",
            max_seq_length=512,
            use_quantization=False,  # FP16 per Priority 1
        ),
        lora=LoRAConfig(rank=16, alpha=32),
        attack=AttackConfig(num_steps=100, learning_rate=1e-4),
        hvp=HVPConfig(num_power_iterations=50, null_space_threshold=1e-6),
    )
