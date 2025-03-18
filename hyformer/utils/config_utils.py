"""Utilities for working with configurations."""

import os
import yaml
from typing import Dict, Any, Optional, Type, TypeVar, Union

from hyformer.configs.base import BaseConfig, ConfigFormat
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.trainer import TrainerConfig

T = TypeVar('T', bound=BaseConfig)

def load_config(
    config_path: str, 
    config_class: Type[T] = BaseConfig,
    format: Optional[ConfigFormat] = None
) -> T:
    """Load a configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        config_class: Configuration class to use
        format: File format (if None, inferred from file extension)
        
    Returns:
        Configuration instance
    """
    return config_class.from_file(config_path, format)

def load_experiment_config(
    config_path: str,
    format: Optional[ConfigFormat] = None
) -> Dict[str, BaseConfig]:
    """Load a complete experiment configuration.
    
    This function loads a configuration file that contains multiple
    configuration sections, such as tokenizer and trainer configurations.
    
    Args:
        config_path: Path to the configuration file
        format: File format (if None, inferred from file extension)
        
    Returns:
        Dictionary mapping configuration names to configuration instances
    """
    # Infer format from file extension if not provided
    if format is None:
        ext = os.path.splitext(config_path)[1].lower()
        if ext == '.json':
            format = ConfigFormat.JSON
        elif ext in ['.yaml', '.yml']:
            format = ConfigFormat.YAML
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    # Load raw configuration
    with open(config_path, 'r') as f:
        if format == ConfigFormat.JSON:
            import json
            raw_config = json.load(f)
        elif format == ConfigFormat.YAML:
            raw_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # Parse configuration sections
    configs = {}
    
    if 'tokenizer' in raw_config:
        configs['tokenizer'] = TokenizerConfig.from_dict(raw_config['tokenizer'])
    
    if 'trainer' in raw_config:
        configs['trainer'] = TrainerConfig.from_dict(raw_config['trainer'])
    
    # Add other configuration sections as needed
    
    return configs

def save_experiment_config(
    configs: Dict[str, BaseConfig],
    config_path: str,
    format: Optional[ConfigFormat] = None
) -> None:
    """Save a complete experiment configuration.
    
    Args:
        configs: Dictionary mapping configuration names to configuration instances
        config_path: Path to save the configuration file
        format: File format (if None, inferred from file extension)
    """
    # Infer format from file extension if not provided
    if format is None:
        ext = os.path.splitext(config_path)[1].lower()
        if ext == '.json':
            format = ConfigFormat.JSON
        elif ext in ['.yaml', '.yml']:
            format = ConfigFormat.YAML
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    # Convert configurations to dictionaries
    raw_config = {name: config.to_dict() for name, config in configs.items()}
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Save configuration
    with open(config_path, 'w') as f:
        if format == ConfigFormat.JSON:
            import json
            json.dump(raw_config, f, indent=2)
        elif format == ConfigFormat.YAML:
            yaml.dump(raw_config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

def print_config_schema(config_class: Type[BaseConfig]) -> None:
    """Print the schema of a configuration class.
    
    Args:
        config_class: Configuration class to print schema for
    """
    schema = config_class.schema()
    
    print(f"{config_class.__name__} Schema:")
    for name, info in schema.items():
        default = info['default']
        default_str = f" (default: {default})" if default is not None else ""
        print(f"  {name}: {info['type']}{default_str}")
        if info['doc']:
            print(f"    {info['doc']}")

# Example usage
if __name__ == "__main__":
    # Print schema
    print_config_schema(TokenizerConfig)
    print()
    print_config_schema(TrainerConfig)
    
    # Load configuration
    configs = load_experiment_config("hyformer/configs/examples/default_config.yaml")
    
    # Access configuration values
    tokenizer_config = configs['tokenizer']
    trainer_config = configs['trainer']
    
    print(f"\nTokenizer: {tokenizer_config.tokenizer}")
    print(f"Vocabulary path: {tokenizer_config.path_to_vocabulary}")
    
    print(f"\nBatch size: {trainer_config.batch_size}")
    print(f"Learning rate: {trainer_config.learning_rate}")
    
    # Modify configuration
    trainer_config.update(batch_size=64, learning_rate=2e-4)
    
    print(f"\nUpdated batch size: {trainer_config.batch_size}")
    print(f"Updated learning rate: {trainer_config.learning_rate}")
    
    # Save configuration
    save_experiment_config(
        configs, 
        "hyformer/configs/examples/modified_config.yaml"
    ) 