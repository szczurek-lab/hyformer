from transformers import PretrainedConfig


class LoggerConfig(PretrainedConfig):

    def __init__(
        self,
        wandb_log: bool = True,
        wandb_project: str = "jointformer",
        wandb_run_name: str = "lm",
        **kwargs,
    ):

        self.wandb_log = wandb_log
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        super().__init__(**kwargs)

    def save(self, save_directory: str) -> None:
        super().save_pretrained(save_directory=save_directory)

    def load(self, config_path: str) -> 'PretrainedConfig':
        return super().from_pretrained(pretrained_model_name_or_path=config_path)
