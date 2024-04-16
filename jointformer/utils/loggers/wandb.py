import os
import json
import wandb

from datetime import datetime


class WandbLogger:

    def __init__(self, logger_config, configs_to_store):

        self.enable_logging = logger_config.wandb_log
        self.project = logger_config.project
        self.name = logger_config.name
        self.configs_to_store = configs_to_store
        self._prepare_config_to_store()
        self.run_id = None

    def init_run(self, run_id=None, resume=False):

        if self.enable_logging:
            if self.run_id and resume:
                print("Resuming run", self.run_id, "...")
                wandb.init(id=self.run_id, resume="must")
            else:
                if self.name is None:
                    self.name = self.configs_to_store['ModelConfig']['model_name']
                self.run = wandb.init(
                    project=self.project,
                    name=self.name,
                    config=self.configs_to_store)
                self.run_id = self.run._run_id

    def log(self, ckpt_log: dict):
        if self.enable_logging:
            wandb.log(ckpt_log)

    def finish(self):
        if self.enable_logging:
            self.run.finish()

    def store_configs(self, out_dir: str):
        with open(os.path.join(out_dir, 'data.json'), 'w') as fp:
            json.dump(self.configs_to_store, fp)

    def _prepare_config_to_store(self):
        output_config = {}
        for config_to_store in self.configs_to_store:
            output_config[str(config_to_store.__class__.__name__)] = config_to_store.to_dict()
        self.configs_to_store = output_config
        return None
