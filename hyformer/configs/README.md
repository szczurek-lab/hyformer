# Configuration Files for Hyformer

This directory contains configuration files for all the components of the Hyformer project. Each configuration file is implemented as a dataclass, inheriting from a common `BaseConfig` class, which provides utility methods for loading, saving, and updating configurations.

## Usage

Each configuration class can be instantiated from a dictionary or a JSON file using the provided class methods. Configurations can be updated dynamically, and changes can be saved back to JSON files for persistence.

### Example Usage
To load a configuration from a JSON file:
```python
from hyformer.configs.dataset import DatasetConfig

# Load configuration from a JSON file
config = DatasetConfig.from_config_filepath('path/to/config.json')

# Access configuration parameters
print(config.dataset_type)
print(config.train_data_path)
```

## Extending Configurations

To extend a configuration, create a new dataclass that inherits from `BaseConfig` and define the necessary parameters. Utilize the utility methods from `BaseConfig` to handle loading, saving, and updating configurations.
