import os
import json
import copy

from typing import Dict, Any


CONFIG_NAME = 'config.json'


class Config:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, x):
        return getattr(self, x)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def save(self, save_directory: str) -> None:
        config_dict = self.to_dict()
        config_path = os.path.join(save_directory, CONFIG_NAME)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        return cls(**config_dict)

    @classmethod
    def from_config_file(cls, config_file_dir: str) -> 'Config':
        config_path = os.path.join(config_file_dir, CONFIG_NAME)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
