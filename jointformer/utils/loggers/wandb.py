import os
import json
import wandb
import time

import torch.nn as nn
import pandas as pd

from rdkit import Chem
from typing import Optional, List

from jointformer.configs.base import Config
from jointformer.configs.logger import LoggerConfig
from jointformer.utils.plot import mol_to_pil_image


class WandbLogger:

    def __init__(
            self,
            enable_logging,
            user,
            project,
            resume,
            watch,
            watch_freq,
            display_name: Optional[str] = None,
            config: Optional[List[Config]] = None
    ):
        self.enable_logging = enable_logging
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

    def store_configs(self, *config_list: List[Config]):
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
        if self.enable_logging:
            self.run = wandb.init(
                entity=self.user, project=self.project, resume=self.resume, name=self.display_name,
                config=self.config, id=self.run_id, reinit=True,
                settings=wandb.Settings(_service_wait=300, start_method="fork")
                )

    def log(self, log: dict):
        if self.enable_logging:
            self.run.log(log)

    def finish(self):
        if self.enable_logging:
            self.run.finish()

    def log_molecule_data(self, data: List[str]) -> None:
        if self.enable_logging:
            data = list(set(data))  # remove duplicates
            out = []
            for smiles in data:
                try:
                    out.extend({
                        'smiles': smiles,
                        'molecule': wandb.Molecule.from_smiles(smiles),
                        'molecule_2d': wandb.Image(mol_to_pil_image(Chem.MolFromSmiles(smiles)))
                    })
                except:
                    pass

            if len(out) > 0:
                dataframe = pd.DataFrame.from_records(data)
                table = wandb.Table(dataframe=dataframe)
                self.run.log(
                    {
                        "table": table,
                        "molecules": [substance.get("molecule") for substance in data],
                    }
                )

    @classmethod
    def from_config(cls, config: LoggerConfig, display_name: str = None):
        display_name = display_name if display_name is not None else config.display_name
        return cls(
            enable_logging=config.enable_logging, user=config.user, project=config.project, resume=config.resume,
            display_name=display_name, watch=config.watch, watch_freq=config.watch_freq
        )
