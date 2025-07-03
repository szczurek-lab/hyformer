import os
import json
import wandb
import time

import torch.nn as nn

from typing import Optional, List

from hyformer.configs.base import BaseConfig
from hyformer.configs.logger import LoggerConfig



class WandbLogger:

    def __init__(
            self,
            enable,
            user,
            project,
            resume,
            watch,
            watch_freq,
            display_name: Optional[str] = None,
            config: Optional[List[BaseConfig]] = None
    ):
        self.enable = enable
        self.user = user
        self.project = project
        self.resume = resume
        self.watch = watch
        self.watch_freq = watch_freq
        self.display_name = None
        self.config = config
        self.run_id = None
        self.run = None

        self.set_run_id()
        self.set_display_name(display_name)

    def set_run_id(self, run_id: Optional[str] = None):
        self.run_id = wandb.util.generate_id() if run_id is None else run_id

    def watch_model(self, model: nn.Module):
        if self.watch:
            self.run.watch(model, log_freq=self.watch_freq, log='all')

    def set_display_name(self, display_name: str = None):
        if display_name is not None:
            self.display_name = display_name
        else:
            try:
                self.display_name = os.environ.get('SLURM_JOB_NAME')
            except KeyError:
                self.display_name = time.strftime("%Y%m%d-%H%M%S")

    def store_configs(self, *config_list: List[BaseConfig]):
        if self.config is None:
            self.config = {}
        for config in config_list:
            config_name = config.__class__.__name__.lower()
            self.config[config_name] = config.to_dict()

    def save_configs(self, out_dir: str):
        if self.config:
            with open(os.path.join(out_dir, 'config.json'), 'w') as fp:
                json.dump(self.config, fp, indent=4)

    def init_run(self):
        if self.enable:
            self.run = wandb.init(
                entity=self.user, project=self.project, resume=self.resume, name=self.display_name,
                config=self.config, id=self.run_id, reinit=True,
                settings=wandb.Settings(_service_wait=300, start_method="fork")
                )

    def log(self, log: dict):
        if self.enable:
            self.run.log(log)

    def finish(self):
        if self.enable:
            self.run.finish()

    @classmethod
    def from_config(cls, config: LoggerConfig, display_name: str = None):
        display_name = display_name if display_name is not None else config.display_name
        return cls(
            enable=config.enable, user=config.user, project=config.project, resume=config.resume,
            display_name=display_name, watch=config.watch, watch_freq=config.watch_freq
        )
