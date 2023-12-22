import torch


def select_random_indices_from_length(length: int, num_indices_to_select: int) -> torch.Tensor:
    return torch.randperm(length)[:num_indices_to_select]
