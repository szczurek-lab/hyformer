import numpy as np

def calculate_perplexity(losses: np.ndarray) -> float:
    """Calculate perplexity from loss values.
    
    Args:
        losses: Array of loss values
        
    Returns:
        Perplexity value
    """
    return np.exp(np.mean(losses)) 