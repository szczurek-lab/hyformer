# Data Utilities for Hyformer

Public API surface (import paths in parentheses):

- **SequenceDataset** (`hyformer.utils.data.datasets.sequence.SequenceDataset`): PyTorch-compatible dataset for sequence tasks.
- **AutoDataset** (`hyformer.utils.data.datasets.auto.AutoDataset`): Factory that builds datasets from `hyformer.configs.dataset.DatasetConfig`.
- **create_dataloader** (`hyformer.utils.data.utils.create_dataloader`): Helper to construct a `torch.utils.data.DataLoader` for a dataset; supports tasks `'lm'`, `'prediction'`, `'mlm'`.

## Quickstart
```python
from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.data.datasets.auto import AutoDataset
from hyformer.utils.data.utils import create_dataloader
from hyformer.tokenizers.base import BaseTokenizer

# 1. Load dataset configuration
config = DatasetConfig.from_config_filepath('path/to/dataset_config.json')

# 2. Create dataset (SequenceDataset under the hood)
dataset = AutoDataset.from_config(config, split='train')

# 3. Initialize tokenizer
#    (must implement BaseTokenizer interface)
tokenizer = BaseTokenizer(...)

# 4. Build DataLoader
loader = create_dataloader(
    dataset=dataset,
    tasks={'lm': 1.0},
    tokenizer=tokenizer,
    batch_size=32,
)
```

For module-level details and alternative storage backends, refer to the docstrings in the `datasets` and `storage` subpackages.
