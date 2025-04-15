import os
import torch
import random
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Global seed for reproducibility
_GLOBAL_SEED = 42


def set_seed(seed: int, use_deterministic_algorithms: bool = False) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    This function sets seeds for Python's random module, NumPy, PyTorch (both CPU and GPU),
    and configures PyTorch's backends for deterministic behavior.
    
    Args:
        seed: Integer seed value to use for all random number generators
        use_deterministic_algorithms: Whether to use deterministic algorithms
    
    Source:
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    
    global _GLOBAL_SEED
    
    if seed is not None:
        _GLOBAL_SEED = seed
        
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Reproducibility settings
    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)  # Use deterministic algorithms where available
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CUDA operations
        torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmarking for reproducibility
        torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for better reproducibility
        torch.backends.cudnn.allow_tf32 = False  # Disable TF32 for better reproducibility
        
        # Set environment variable for CUBLAS workspace config (for CUDA 10.2+)
        # This ensures deterministic behavior in CUBLAS operations by:
        # 1. Setting a fixed workspace size (4096 bytes)
        # 2. Allocating a fixed number of workspace buffers (8)
        # Without this, CUBLAS may use different workspace sizes across runs,
        # leading to non-deterministic results in matrix operations.
        if torch.cuda.is_available():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")
    return seed


def get_global_seed():
    """Get the current global seed.
    
    Returns:
        The current global seed value
    """
    global _GLOBAL_SEED
    return _GLOBAL_SEED


def seed_worker(worker_id):
    """Set seed for each worker to ensure reproducibility.
    
    This function is used as a worker_init_fn in PyTorch DataLoader to ensure
    that each worker has a different but deterministic seed. It sets seeds for
    NumPy, Python's random module, and PyTorch.
    
    Args:
        worker_id: ID of the worker process
    """
    global _GLOBAL_SEED
    
    # Use the global seed as the base, but each worker gets a different seed
    # derived from the global seed and the worker ID
    base_seed = _GLOBAL_SEED
    worker_seed = base_seed + worker_id
    
    # Ensure the worker seed is a 32-bit integer
    worker_seed = worker_seed % 2**32
    
    # Set seeds for worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    