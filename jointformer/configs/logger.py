from typing import Optional, List
from jointformer.configs.base import Config


class LoggerConfig(Config):

    def __init__(
        self,
        log,
        user,
        project,
        resume,
        display_name: Optional[str] = None,
        config: Optional[List[Config]] = None,
    ):

        super().__init__()
        self.log = log
        self.user = user
        self.project = project
        self.resume = resume
        self.display_name = display_name
        self.config = config
