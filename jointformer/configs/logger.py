from typing import Optional, List
from hyformer.configs.base import BaseConfig


class LoggerConfig(BaseConfig):

    def __init__(
        self,
        logger_name: str,
        enable_logging: bool,
        user,
        project,
        resume,
        watch,
        watch_freq,
        display_name: Optional[str] = None,
        config: Optional[List[BaseConfig]] = None,
    ):

        super().__init__()
        self.logger_name = logger_name
        self.enable_logging = enable_logging
        self.user = user
        self.project = project
        self.resume = resume
        self.watch = watch
        self.watch_freq = watch_freq
        self.display_name = display_name
        self.config = config
