from transformers import PretrainedConfig
from typing import List

from hybrid_transformer.utils.datasets.guacamol import GuacamolSMILESDataset

import importlib


class LoggerConfig(PretrainedConfig):
    task_type = "distribution_learning"

    def __init__(
        self,
        **kwargs,
    ):
        # wandb loggers
        wandb_log = False  # disabled by default
        wandb_project = 'owt'
        wandb_run_name = 'gpt2'  # 'run' + str(time.time())

        super().__init__(**kwargs)

    def save(self, save_directory: str) -> None:
        super().save_pretrained(save_directory=save_directory)

    def load(self, config_path: str) -> 'PretrainedConfig':
        return super().from_pretrained(pretrained_model_name_or_path=config_path)
