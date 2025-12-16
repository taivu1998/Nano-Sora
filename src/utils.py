import os
import random
import logging
import numpy as np
import torch
from typing import Tuple, Optional

def seed_everything(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Configures a logger to output to console and file.
    Prevents duplicate handlers when called multiple times.
    """
    logger = logging.getLogger(name)

    # Clear existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def patchify(x: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Reshapes (B, C, T, H, W) video tensor into (B, N, Patch_Vol) tokens.
    Crucial for calculating the flow matching loss in patch space.

    Args:
        x: Input tensor of shape (B, C, T, H, W)
        patch_size: Tuple of (pt, ph, pw) - patch sizes for time, height, width

    Returns:
        Tensor of shape (B, N, C*pt*ph*pw) where N = (T/pt) * (H/ph) * (W/pw)

    Raises:
        ValueError: If input dimensions are not divisible by patch sizes
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B, C, T, H, W), got {x.dim()}D")

    B, C, T, H, W = x.shape
    pt, ph, pw = patch_size

    # Validate divisibility
    if T % pt != 0:
        raise ValueError(f"Temporal dimension {T} not divisible by patch size {pt}")
    if H % ph != 0:
        raise ValueError(f"Height {H} not divisible by patch size {ph}")
    if W % pw != 0:
        raise ValueError(f"Width {W} not divisible by patch size {pw}")

    # 1. Reshape into grid of patches
    x = x.reshape(B, C, T//pt, pt, H//ph, ph, W//pw, pw)

    # 2. Permute to group patch content: (B, T_grid, H_grid, W_grid, C, pt, ph, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)

    # 3. Flatten to (B, N, Patch_Vol)
    x = x.flatten(1, 3).flatten(2)
    return x

def unpatchify(x: torch.Tensor, patch_size: Tuple[int, int, int], out_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
    """
    Reconstructs video tensor from patch tokens.

    Args:
        x: Input tensor of shape (B, N, Patch_Vol)
        patch_size: Tuple of (pt, ph, pw) - patch sizes for time, height, width
        out_shape: Target shape (B, C, T, H, W)

    Returns:
        Tensor of shape (B, C, T, H, W)

    Raises:
        ValueError: If shapes are incompatible
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor (B, N, Patch_Vol), got {x.dim()}D")

    B, C, T, H, W = out_shape
    pt, ph, pw = patch_size

    # Validate shapes
    T_grid = T // pt
    H_grid = H // ph
    W_grid = W // pw
    expected_N = T_grid * H_grid * W_grid
    expected_patch_vol = C * pt * ph * pw

    if x.shape[1] != expected_N:
        raise ValueError(f"Expected N={expected_N} tokens, got {x.shape[1]}")
    if x.shape[2] != expected_patch_vol:
        raise ValueError(f"Expected patch volume={expected_patch_vol}, got {x.shape[2]}")

    # 1. Reshape flat tokens back to grid
    x = x.view(B, T_grid, H_grid, W_grid, C, pt, ph, pw)

    # 2. Permute back to spatial structure
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)

    # 3. Fuse dimensions
    x = x.reshape(B, C, T, H, W)
    return x

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
