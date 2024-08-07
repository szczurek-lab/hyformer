import torch


class TorchConvertMixin:

    def to_tensor(self, targets):
        return torch.from_numpy(targets)
    