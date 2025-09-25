# Hyformer Models Overview

This package contains the Hyformer architecture, supporting modules, and a
minimal factory for constructing models.

## Key files

- `auto.py` – simple `AutoModel` factory (`if`/`elif` dispatch, no registry)
- `hyformer.py` – core Hyformer implementation
- `backbone.py`, `layers/`, `utils/` – reusable building blocks and helpers
- `wrappers/` – inference utilities for generation or encoding workflows

## Basic usage

```python
from hyformer import AutoModel

# Load weights from disk or the Hugging Face Hub
model = AutoModel.from_pretrained("SzczurekLab/hyformer_peptides")
```
