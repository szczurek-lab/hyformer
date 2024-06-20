import os, sys
import torch
import logging
import random
import json

import numpy as np

from collections.abc import MutableMapping


logger = logging.getLogger(__name__)


def dump_configs(out_dir: str, *config_list):
    configs_to_store = {}
    for config in config_list:
        if config is not None:
            config_name = config.__class__.__name__.lower()
            configs_to_store[config_name] = config.to_dict()

    with open(os.path.join(out_dir, 'config.json'), 'w') as fp:
        json.dump(configs_to_store, fp, indent=4)


def create_output_dir(out_dir):
    if not os.path.isdir(out_dir):
        is_ddp = int(os.environ.get('RANK', -1)) != -1
        is_master_process = int(os.environ.get('RANK', -1)) == 0
        if (is_master_process and is_ddp) or not is_ddp:
            os.makedirs(out_dir, exist_ok=False)
            logger.info(f"Output directory {out_dir} created...")
        
        
def log_args(args):
    logging.info("Logging experiment...")
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)


def set_to_dev_mode(**kwargs):
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
        if hasattr(model_config, "num_heads"):
            model_config.num_heads = 1
        if hasattr(model_config, "embedding_dim"):
            model_config.embedding_dim = 16
    if trainer_config and hasattr(trainer_config, "batch_size"):
        trainer_config.batch_size = 2
        trainer_config.max_iters = 400
        trainer_config.eval_every = 100
        trainer_config.eval_iters = 2
    if logger_config and hasattr(logger_config, "enable_wandb"):
        logger_config.display_name = 'test'


def set_seed(seed: int = 42) -> None:
    """
    Source:
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")
    return None


def get_device() -> torch.device or str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_strings_to_file(strings, filename):
    with open(filename, 'w') as f:
        for s in strings:
            f.write(s + '\n')


def read_strings_from_file(filename):
    with open(filename, 'r') as f:
        strings = f.read().splitlines()
    return strings


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
