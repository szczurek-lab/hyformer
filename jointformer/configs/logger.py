from jointformer.configs.base import Config


class LoggerConfig(Config):

    def __init__(
        self,
        wandb_log,
        wandb_project,
        wandb_run_name
    ):

        super().__init__()
        self.wandb_log = wandb_log
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
