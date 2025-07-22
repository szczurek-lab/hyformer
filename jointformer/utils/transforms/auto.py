import importlib

from torchvision import transforms
from typing import Any, List


class AutoTransform:

    @classmethod
    def from_config(cls, config: List) -> Any:

        if config is None:
            return None
        else:
            transform = []
            for transform_config in config:

                if transform_config['name'] == 'smiles_enumerator':
                    transform.append(getattr(importlib.import_module(
                        "hyformer.utils.transforms.enumerator"),
                        "SmilesEnumerator").from_config(transform_config['params']))

            return transforms.Compose(transform)


class AutoTargetTransform:

    @classmethod
    def from_config(cls, config: dict) -> Any:
        
        if config['name'] == 'scaler':
            return getattr(importlib.import_module(
                "hyformer.utils.transforms.scaler"),
                "Scaler").from_config(config['params'])
        
        elif config['name'] == 'scaler_test_time_only':
            return getattr(importlib.import_module(
                "hyformer.utils.transforms.scaler"),
                "ScalerTestTimeOnly").from_config(config['params'])
    
        else:
            raise ValueError(f"Target transform {config['name']} not supported.")
        