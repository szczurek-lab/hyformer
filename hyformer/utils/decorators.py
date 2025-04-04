import torch
import functools


def inference(fn):
    """
    Decorator to wrap model inference:
    - Disables gradient computation
    - Sets model to eval mode (and restores previous mode afterward)
    - Optionally moves inputs to the model's device (if 'self' has .device or .module.device)
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        model = args[0]  # assumes method is bound to `self`
        
        # Determine model module (support for DDP/wrappers)
        module = getattr(model, 'module', model)
        
        # Save original mode and switch to eval
        was_training = module.training
        module.eval()

        # Disable grad and run inference
        with torch.no_grad():
            result = fn(*args, **kwargs)

        # Restore training mode if needed
        if was_training:
            module.train()

        return result

    return wrapper
