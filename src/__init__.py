"""
NeuroEntropy: The Thermodynamics of Intelligence Loss in LLMs

A research laboratory for studying metabolic attacks on large language models.
"""

__version__ = "0.1.0"

from .catalyst import HessianAwareCatalyst
from .diagnosis import (
    compute_effective_rank,
    compute_spectral_gap,
    diagnose_model_health,
    ActivationHook
)
from .attack_loop import MetabolicAttackLoop
from .lora_attack import LoRAPoisoning
from .config import (
    ExperimentConfig,
    ModelConfig,
    LoRAConfig,
    AttackConfig,
    HVPConfig,
    get_pythia_70m_config,
    get_pythia_160m_config,
    get_pythia_410m_config
)

__all__ = [
    'HessianAwareCatalyst',
    'compute_effective_rank',
    'compute_spectral_gap',
    'diagnose_model_health',
    'ActivationHook',
    'MetabolicAttackLoop',
    'LoRAPoisoning',
    'ExperimentConfig',
    'ModelConfig',
    'LoRAConfig',
    'AttackConfig',
    'HVPConfig',
    'get_pythia_70m_config',
    'get_pythia_160m_config',
    'get_pythia_410m_config',
]
