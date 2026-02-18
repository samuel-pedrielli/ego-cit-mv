"""
transformer_cit package

Lightweight skeleton implementation for CIT-on-transformers (monitoring + ablation plumbing).
See: spec.md
"""

from .model import CITModel
from .critics import CriticEnsemble
from .losses import IdentityLoss, IdentityStabilityLoss, WelfareLoss, CITLoss
from .schedule import FAPConfig, ForgeAnchorPreserve

__all__ = [
    "CITModel",
    "CriticEnsemble",
    "IdentityLoss",
    "IdentityStabilityLoss",
    "WelfareLoss",
    "CITLoss",
    "FAPConfig",
    "ForgeAnchorPreserve",
]
