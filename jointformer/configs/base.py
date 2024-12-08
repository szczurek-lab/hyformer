import os
import json
import copy
from typing_extensions import Self
from typing import Dict, Any


class BaseConfig:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, x):
        return getattr(self, x)
    
    def __setitem__(self, x, y):
        setattr(self, x, y)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def save(self, filename: str) -> None:
        config_dict = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Self:
        return cls(**config_dict)

    @classmethod
    def from_config_file(cls, config_filepath: str) -> Self:
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Configuration file {config_filepath} not found.")
        with open(config_filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
