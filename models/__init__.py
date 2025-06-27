"""
CLIP++ Models Package

Model implementations for CLIP++.
"""

import torch
import warnings
from pkg_resources import packaging
from .base import Base_PromptLearner
from .clip import build_model

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

__all__ = [
    "Base_PromptLearner",
    "build_model",
]