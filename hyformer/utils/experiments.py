import os, sys
import logging
import json
from typing import Any, List, Optional, Set

import numpy as np
import torch

logger = logging.getLogger(__name__)


def dump_configs(out_dir: str, *config_list) -> None:
    configs_to_store = {}
    for config in config_list:
        if config is not None:
            config_name = config.__class__.__name__.lower()
            configs_to_store[config_name] = config.to_dict()

    with open(os.path.join(out_dir, "config.json"), "w") as fp:
        json.dump(configs_to_store, fp, indent=4)


def log_args(args) -> None:
    logging.info("Logging experiment...")
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)


def set_to_dev_mode(**kwargs) -> None:
    logger.info("Dev mode is on")
    task_config = kwargs.get("task_config", None)
    model_config = kwargs.get("model_config", None)
    trainer_config = kwargs.get("trainer_config", None)
    logger_config = kwargs.get("logger_config", None)

    if task_config:
        if hasattr(task_config, "num_samples"):
            task_config.num_samples = 4
    if model_config:
        if hasattr(model_config, "num_layers"):
            model_config.num_layers = 1
        if hasattr(model_config, "num_attention_heads"):
            model_config.num_attention_heads = 1
        if hasattr(model_config, "embedding_dim"):
            model_config.embedding_dim = 16
    if trainer_config and hasattr(trainer_config, "batch_size"):
        trainer_config.batch_size = 2
        trainer_config.max_iters = 400
        trainer_config.eval_every = 100
        trainer_config.eval_iters = 2
    if logger_config and hasattr(logger_config, "enable_wandb"):
        logger_config.display_name = "test"


def get_device() -> torch.device | str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_strings_to_file(strings: List[str], filename: str) -> None:
    with open(filename, "w") as f:
        for s in strings:
            f.write(s + "\n")


def read_strings_from_file(filename: str) -> List[str]:
    with open(filename, "r") as f:
        strings = f.read().splitlines()
    return strings


def find_multiple(value: int, multiple_of: int) -> int:
    """Find the nearest multiple of a given value."""
    return int(np.ceil(value / multiple_of) * multiple_of)


# ---- Merged file I/O helpers from file_io.py ----


def save_json(data: Any, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def write_dict_to_file(dictionary: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4)


def remove_duplicates(list_with_duplicates: List[Any]) -> List[Any]:
    unique_set: Set[Any] = set()
    unique_list: List[Any] = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)
    return unique_list


def get_random_subset(
    dataset: List[Any], subset_size: int, seed: Optional[int] = None
) -> List[Any]:
    if len(dataset) < subset_size:
        raise Exception(
            f"The dataset to extract a subset from is too small: {len(dataset)} < {subset_size}"
        )
    rng_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    subset = np.random.choice(dataset, subset_size, replace=False)
    if seed is not None:
        np.random.set_state(rng_state)
    return list(subset)


def load_npy_with_progress(
    filepath: str,
    mmap_mode: Optional[str] = "r",
    chunk_size: int = 1000,
    show_progress: bool = True,
) -> np.ndarray:
    try:
        array = np.load(filepath, mmap_mode=mmap_mode)
    except ValueError:
        array = np.load(filepath, allow_pickle=True)
    if not show_progress:
        return array
    loaded_array = np.empty_like(array)
    n_chunks = (len(array) + chunk_size - 1) // chunk_size
    from tqdm import tqdm

    with tqdm(total=len(array), desc=f"Loading {os.path.basename(filepath)}") as pbar:
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(array))
            loaded_array[start:end] = array[start:end]
            pbar.update(end - start)
    return loaded_array
