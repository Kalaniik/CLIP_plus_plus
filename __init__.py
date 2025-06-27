"""
CLIP++ Package

A PyTorch implementation of CLIP with enhanced prompt learning capabilities.
"""

from .clip_plus_plus import ClipPlusPlus, PromptLearner

__version__ = "1.0.0"
__author__ = "Guo Kai"

__all__ = [
    "ClipPlusPlus",
    "PromptLearner",
]