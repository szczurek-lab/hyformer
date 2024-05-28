""" A collection of utility functions for data-related tasks.

Source:
    [1] https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/chemistry.py
"""

import json

from typing import Any, Set, List, Optional

import numpy as np


def save_strings_to_file(strings, filename):
    with open(filename, 'w') as f:
        for s in strings:
            f.write(s + '\n')


def read_strings_from_file(filename):
    with open(filename, 'r') as f:
        strings = f.read().splitlines()
    return strings


def write_dict_to_file(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)


def remove_duplicates(list_with_duplicates):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.

    Args:
        list_with_duplicates: list that possibly contains duplicates

    Returns:
        A list with no duplicates.
    """

    unique_set: Set[Any] = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list


def get_random_subset(dataset: List[Any], subset_size: int, seed: Optional[int] = None) -> List[Any]:
    """
    Get a random subset of some dataset.

    For reproducibility, the random number generator seed can be specified.
    Nevertheless, the state of the random number generator is restored to avoid side effects.

    Args:
        dataset: full set to select a subset from
        subset_size: target size of the subset
        seed: random number generator seed. Defaults to not setting the seed.

    Returns:
        subset of the original dataset as a list
    """
    if len(dataset) < subset_size:
        raise Exception(f'The dataset to extract a subset from is too small: '
                        f'{len(dataset)} < {subset_size}')

    # save random number generator state
    rng_state = np.random.get_state()

    if seed is not None:
        # extract a subset (for a given training set, the subset will always be identical).
        np.random.seed(seed)

    subset = np.random.choice(dataset, subset_size, replace=False)

    if seed is not None:
        # reset random number generator state, only if needed
        np.random.set_state(rng_state)

    return list(subset)
