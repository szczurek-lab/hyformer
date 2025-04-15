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


### .lmdb file handling ###

import lmdb
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm
from functools import partial


def process_keys(lmdb_path, max_readers, keys_to_extract, key_subset):
    """
    Worker function that opens the LMDB environment,
    retrieves specified fields from every entry in key_subset,
    and returns a list of string values for the first key in keys_to_extract.
    
    Args:
        lmdb_path: Path to the LMDB file
        max_readers: Maximum number of readers for the LMDB environment
        keys_to_extract: List of field names to extract from each entry
        key_subset: List of keys to process
    
    Returns:
        List of string values extracted from the first key in keys_to_extract
    """
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers,
    )
    results = []
    with env.begin() as txn:
        for key in key_subset:
            datapoint_pickled = txn.get(key)
            data = pickle.loads(datapoint_pickled)
            # Extract the string value for the first key in keys_to_extract
            if keys_to_extract[0] in data:
                results.append(str(data[keys_to_extract[0]]))
    env.close()
    return results


def load_lmdb_file(lmdb_path, max_readers=256, keys_to_extract=['smi'], num_workers=8):
    """
    Load an LMDB file and extract string values for a specified field.
    
    Args:
        lmdb_path: Path to the LMDB file
        max_readers: Maximum number of readers for the LMDB environment
        keys_to_extract: List of field names to extract (only the first key is used)
        num_workers: Number of worker processes for parallel processing
    
    Returns:
        List of string values for the first key in keys_to_extract
    """
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers,
    )
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    env.close()

    # Calculate an appropriate chunk size based on the total number of keys.
    chunk_size = len(keys) // num_workers + 1

    # Partition keys into roughly equal chunks.
    key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

    # Create a pool of worker processes.
    with multiprocessing.Pool(num_workers) as pool:
        # Initialize tqdm progress bar with the total number of keys.
        pbar = tqdm(total=len(keys), desc="Processing LMDB entries")

        results_list = []
        # Use imap_unordered to process chunks and update progress as each chunk is done.
        for chunk_result in pool.imap_unordered(partial(process_keys, lmdb_path, max_readers, keys_to_extract), key_chunks):
            results_list.append(chunk_result)
            # Update progress bar with the number of keys processed in this chunk.
            pbar.update(len(chunk_result))
        pbar.close()

    # Flatten the list of lists into a single list of string values.
    all_data = [item for sublist in results_list for item in sublist]

    return all_data


### Handling string-like data ###


from typing import List


def infer_string_dtype(strings: List[str], buffer_percent: float = 20.0) -> str:
    """
    Infer the optimal string dtype length from a list of strings.
    
    Args:
        strings: List of strings to analyze
        buffer_percent: Percentage of buffer to add to max length
        
    Returns:
        Optimal numpy dtype string (e.g., '<U100')
    """
    if isinstance(strings, np.ndarray):
        if strings.size == 0:
            return '<U1'
        # Convert numpy array to list of strings
        strings = strings.tolist()
    elif not strings:  # Empty list
        return '<U1'
    
    # Get maximum length
    max_len = max(len(s) for s in strings)
    
    # Add buffer
    buffer = int(max_len * (buffer_percent / 100))
    optimal_len = max_len + buffer
    
    # Round up to nearest 10 for cleaner numbers
    optimal_len = ((optimal_len + 9) // 10) * 10
    
    return f'<U{optimal_len}'


### Handling numpy arrays ###

import os

def load_npy_with_progress(
    filepath: str,
    mmap_mode: Optional[str] = 'r',
    chunk_size: int = 1000,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Load a .npy file with progress bar.
    
    Args:
        filepath: Path to the .npy file
        mmap_mode: Memory map mode ('r' for read-only)
        chunk_size: Number of items to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        Loaded numpy array
    """
    try:
        # First try to load with memory mapping
        array = np.load(filepath, mmap_mode=mmap_mode)
    except ValueError:
        # If memory mapping fails (e.g., for string arrays), load normally
        array = np.load(filepath, allow_pickle=True)
    
    # If not showing progress, return the array
    if not show_progress:
        return array
    
    # Create a new array to store the loaded data
    loaded_array = np.empty_like(array)
    
    # Calculate number of chunks
    n_chunks = (len(array) + chunk_size - 1) // chunk_size
    
    # Load data in chunks with progress bar
    with tqdm(total=len(array), desc=f"Loading {os.path.basename(filepath)}") as pbar:
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(array))
            loaded_array[start:end] = array[start:end]
            pbar.update(end - start)
    
    return loaded_array
