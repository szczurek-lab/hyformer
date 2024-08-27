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
                        "jointformer.utils.transforms.smiles.enumerate"),
                        "SmilesEnumerator").from_config(transform_config['params']))

            return transforms.Compose(transform)
