import os
import json
import wandb

from datetime import datetime
from typing import Optional, List
from jointformer.configs.base import Config
from jointformer.configs.logger import LoggerConfig


class WandbLogger:

    def __init__(
            self,
            enable_logging,
            user,
            project,
            resume,
            display_name: Optional[str] = None,
            config: Optional[List[Config]] = None
    ):
        self.enable_logging = enable_logging
        self.user = user
        self.project = project
        self.resume = resume
        self.display_name = display_name
        self.config = config

    def set_display_name(self, display_name: str):
        self.display_name = display_name

    def store_config(self, *config_list: List[Config]):
        if self.config is None:
            self.config = {}
        for config in config_list:
            config_name = config.__class__.__name__.lower()
            self.config[config_name] = config.to_dict()

    def save_config(self, out_dir: str):
        if self.config:
            with open(os.path.join(out_dir, 'config.json'), 'w') as fp:
                json.dump(self.config, fp, indent=4)

    def init_run(self):
        if self.enable_logging:
            wandb.init(
                entity=self.user, project=self.project, resume=self.resume, name=self.display_name, config=self.config)

    def log(self, log: dict):
        if self.enable_logging:
            wandb.log(log)

    @classmethod
    def from_config(cls, config: LoggerConfig, display_name: str = None):
        display_name = display_name if display_name is not None else config.display_name
        return cls(
            enable_logging=config.enable_logging, user=config.user, project=config.project, resume=config.resume,
            display_name=display_name
        )
