"""
Global seed management utilities for reproducible experiments.

This module provides a centralized way to set random seeds across all
random number generators used in the platform (torch, numpy, scikit-learn, random).
"""

import os
import time
import random
import numpy as np
import torch
import logging
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

def set_cublas_workspace_config(logger: Optional[logging.Logger] = None):
    """Set CUBLAS workspace config based on available GPU memory.

    Required for CUDA >= 10.2, otherwise torch.use_deterministic_algorithms will throw an error.
    """
    try:
        if torch.cuda.is_available():
            # Get total GPU memory in GB
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Determine workspace size based on GPU memory
            if gpu_memory_gb >= 40:  
                workspace_config = ":4096:8"
            elif gpu_memory_gb >= 20:  
                workspace_config = ":2048:8"  
            elif gpu_memory_gb >= 10:  
                workspace_config = ":1024:8"
            else:  
                workspace_config = ":512:8"
                
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", workspace_config)
            if logger:
                logger.info(f"Set CUBLAS workspace config to {workspace_config} (GPU: {gpu_memory_gb:.1f}GB)")
        else:
            # CPU only, set a minimal workspace config
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            if logger:
                logger.info("Set minimal CUBLAS workspace config for CPU")
                
    except Exception as e:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        if logger:
            logger.warning(f"Could not detect GPU memory, using fallback config: {e}")

def seed_worker(worker_id: int):
    """Use with torch.utils.data.DataLoader(worker_init_fn=seed_worker) to sync NumPy/random per-worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dataloader_generator(seed: int) -> "torch.Generator":
    """
    Use with torch.utils.data.DataLoader(generator=dataloader_generator(seed)) to sync NumPy/random per-worker.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def set_global_seed(seed: Optional[int] = None,
                    logger: Optional[logging.Logger] = None,
                    deterministic: bool = False
                    ) -> int:
    """
    Set the global random seed for all random number generators.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch
    
    Args:
        seed: The seed value to use. If None, uses current timestamp.
        logger: Optional logger to log the seed value.
    
    Returns:
        The seed value that was set.
    """
    global _GLOBAL_SEED
    
    # Generate seed from current time if not provided
    if seed is None:
        seed = int(time.time() * 1000000) % (2**31)
    
    # Store the global seed
    _GLOBAL_SEED = seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    if deterministic:
        # Set deterministic behavior for reproducibility
        # Note: This can significantly slow down operations. Only use if you need to be 100% reproducible
        
        # cuBLAS-based operations
        set_cublas_workspace_config(logger)
        # cuDNN-based operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        # CUDA/cuBLAS-based operations
        torch.backends.cuda.matmul.allow_tf32 = False

        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception as e:
                print(f'torch.use_deterministic_algorithms is not available: {e}')
    
    if logger:
        logger.info(f"Global random seed set to: {seed}")
        logger.info(f"Deterministic: {deterministic}")
        logger.info(f"Use TF32: {torch.backends.cuda.matmul.allow_tf32}")
        logger.info(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        logger.info(f"cuDNN Allow TF32: {torch.backends.cudnn.allow_tf32}")
        logger.info(f"cuBLAS Workspace Config: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    return seed


def get_sklearn_random_state(use_global: bool = True) -> Optional[int]:
    """
    Get a random state value suitable for scikit-learn models.
    
    Args:
        use_global: If True, returns the global seed. If False, returns None.
    
    Returns:
        The seed value to use for scikit-learn's random_state parameter.
    """
    if use_global and _GLOBAL_SEED is not None:
        return _GLOBAL_SEED
    return None