import os
import json
import copy
from typing import Dict, Any
from typing_extensions import Self


class BaseConfig:
    """ Base configuration class for all configuration classes. """

    def __init__(self, **kwargs):
        """ Initialize the configuration class from a dictionary of keyword arguments. """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, x):
        """ Get an attribute of the configuration class. """
        return getattr(self, x)
    
    def __setitem__(self, x, y):
        """ Set an attribute of the configuration class. """
        setattr(self, x, y)

    def __repr__(self):
        """ Return a string representation of the configuration class. """
        return f"{self.__class__.__name__}({self.__dict__})"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Self:
        """ Create a configuration class from a dictionary. """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """ Convert the configuration class to a dictionary. """
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_config_file(cls, config_filepath: str) -> Self:
        """ Create a configuration class from a JSON file. """
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Configuration file {config_filepath} not found.")
        with open(config_filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save(self, filename: str) -> None:
        """ Save the configuration class to a JSON file. """
        config_dict = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
