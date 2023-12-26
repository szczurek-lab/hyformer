import torch
from collections.abc import MutableMapping


def select_random_indices_from_length(length: int, num_indices_to_select: int) -> torch.Tensor:
    return torch.randperm(length)[:num_indices_to_select]


def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
