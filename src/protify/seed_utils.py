"""
Global seed management utilities for reproducible experiments.

This module provides a centralized way to set random seeds across all
random number generators used in the platform (torch, numpy, scikit-learn, random).
"""

import os
import time
import random
import numpy as np
from typing import Optional

# Global variable to store the current seed
_GLOBAL_SEED: Optional[int] = None


def get_global_seed() -> Optional[int]:
    """
    Get the currently set global seed.
    
    Returns:
        The current global seed value, or None if not set.
    """
    return _GLOBAL_SEED


def set_cublas_workspace_config() -> None:
    """Set CUBLAS workspace config to an allowed deterministic value.

    Must be set BEFORE importing torch. Valid values (per NVIDIA docs):
      - ":4096:8" (recommended)
      - ":16:8"   (minimal workspace)
    """
    # Only set if not already provided by the environment/user
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """Use with torch.utils.data.DataLoader(worker_init_fn=seed_worker) to sync NumPy/random per-worker."""
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: Optional[int]):
    """
    Use with torch.utils.data.DataLoader(generator=dataloader_generator(seed)) to sync NumPy/random per-worker.
    """
    import torch
    
    if seed is None:
        seed = set_global_seed()

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set the global random seed for all random number generators.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch
    
    Args:
        seed: The seed value to use. If None, uses current timestamp.
    
    Returns:
        The seed value that was set.
    """    
    # Generate seed from current time if not provided
    if seed is None:
        seed = int(time.time() * 1000000) % (2**31)
    
    # Store the global seed
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
    random.seed(seed)
    np.random.seed(seed)

    # Import torch lazily to avoid initializing CUDA before env is set elsewhere
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    return seed


def set_determinism() -> None:
    # set_cublas_workspace_config() must happen BEFORE importing torch
    #set_cublas_workspace_config()

    # Import torch only after the env var has been set
    import torch

    # Set deterministic behavior for reproducibility
    # Note: This can significantly slow down operations. Only use if you need to be 100% reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception as e:
            print(f'torch.use_deterministic_algorithms is not available: {e}')
            # print torch version
            print(f'torch version: {torch.__version__}')
            print('Make sure you are using the correct version of torch')
