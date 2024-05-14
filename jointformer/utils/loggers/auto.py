import importlib

from typing import Union, Optional
from jointformer.configs.logger import LoggerConfig


class AutoLogger:

    @classmethod
    def from_config(cls, config: LoggerConfig, display_name: Optional[str] = None) -> 'WandbLogger':

        display_name = display_name if display_name is not None else config.display_name

        if config.logger_name == 'wandb':
            return getattr(importlib.import_module(
                "jointformer.utils.loggers.wandb"),
                "WandbLogger").from_config(config, display_name)
        else:
            raise ValueError(f"Logger {config.logger_name} not available.")
