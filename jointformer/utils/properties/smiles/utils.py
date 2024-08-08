try:
    import torch
except:
    torch = None

class TorchConvertMixin:

    def to_tensor(self, targets):
        return torch.from_numpy(targets)
    