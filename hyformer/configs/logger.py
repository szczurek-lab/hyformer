from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from hyformer.configs.base import BaseConfig


@dataclass
class LoggerConfig(BaseConfig):
    """Configuration for logging functionality. """
    logger_type: str  # Name of the logger to use (e.g., 'wandb', 'tensorboard')
    enable: bool  # Whether logging is enabled
    user: str  # Username for the logging service
    project: str  # Project name for the logging service
    resume: bool  # Whether to resume a previous run
    watch: bool  # Whether to watch model parameters
    watch_freq: int  # Frequency of watching model parameters
    display_name: Optional[str] = None  # Display name for the run
    config: Optional[Dict[str, Any]] = None  # Additional configuration parameters for the logger
