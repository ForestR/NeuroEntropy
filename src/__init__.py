"""
NeuroEntropy: The Thermodynamics of Intelligence Loss in LLMs

A research laboratory for studying metabolic attacks on large language models.
"""

__version__ = "0.1.0"

from .catalyst import HessianAwareCatalyst
from .diagnosis import compute_effective_rank, compute_spectral_gap, diagnose_model_health
from .attack_loop import MetabolicAttackLoop

__all__ = [
    'HessianAwareCatalyst',
    'compute_effective_rank',
    'compute_spectral_gap',
    'diagnose_model_health',
    'MetabolicAttackLoop',
]
