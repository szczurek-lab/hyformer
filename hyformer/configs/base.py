import os
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class BaseConfig:
    """Base configuration class for all configuration classes. """

    def __repr__(self):
        """Return a string representation of the configuration class."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"{self.__class__.__name__}({attrs})"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a configuration class from a dictionary."""
        # Check for unknown parameters
        unknown_params = set(config_dict.keys()) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown_params:
            raise ValueError(f"Parameters not specified in the config file: {unknown_params}.")
        
        # Create instance with parameters
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration class to a dictionary."""
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}

    @classmethod
    def from_config_file(cls, config_filepath: str):
        """Create a configuration class from a JSON file."""
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Configuration file {config_filepath} not found.")
        with open(config_filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save(self, filename: str) -> None:
        """Save the configuration class to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        config_dict = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    def update(self, **kwargs):
        """Update configuration parameters and return self for chaining.
        
        This method will only update parameters that are defined in the dataclass.
        Any parameters not defined in the dataclass will be silently ignored.
        
        Args:
            **kwargs: Keyword arguments to update the configuration with.
            
        Returns:
            self: The updated configuration object.
        """
        # Get the fields defined in the dataclass
        defined_fields = {f.name for f in self.__class__.__dataclass_fields__.values()}
        
        # Only update parameters that are defined in the dataclass
        for k, v in kwargs.items():
            if k in defined_fields:
                setattr(self, k, v)
        
        return self
