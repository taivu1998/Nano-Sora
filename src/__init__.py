"""
Nano-Sora: A minimal Diffusion Transformer for Video Generation using Rectified Flow.

This package implements the core components of a video diffusion model:
- 3D Spacetime Patch Embedding (tubelets)
- Diffusion Transformer (DiT) with AdaLN-Zero conditioning
- Rectified Flow for efficient training and sampling
"""

from .model import NanoSora, PatchEmbed3D, DiTBlock, TimestepEmbedder
from .utils import patchify, unpatchify, seed_everything, get_logger
from .dataset import MovingMNIST
from .trainer import Trainer
from .config_parser import load_config, parse_args, merge_config

__version__ = "0.1.0"

__all__ = [
    # Model components
    "NanoSora",
    "PatchEmbed3D",
    "DiTBlock",
    "TimestepEmbedder",
    # Utilities
    "patchify",
    "unpatchify",
    "seed_everything",
    "get_logger",
    # Dataset
    "MovingMNIST",
    # Training
    "Trainer",
    # Config
    "load_config",
    "parse_args",
    "merge_config",
]
