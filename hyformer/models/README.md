# Hyformer Models Directory

This directory contains all model architectures, components, utilities, and wrappers for the Hyformer project. 

## Directory Structure

```
models/
│
├── __init__.py                # Exposes main models and registry for easy import
│
├── auto.py                    # Model registry/factory (AutoModel)
│
├── core/                      # Core model abstractions and main Hyformer model
│   ├── __init__.py
│   ├── base.py                # Base classes (PreTrainedModel, EncoderModel, etc.)
│   ├── hyformer.py            # Main Hyformer model class
│   └── llama_backbone.py      # Transformer backbone for Hyformer
│
├── baselines/                 # Baseline and reference model implementations
│   ├── chemberta.py
│   ├── gpt.py
│   ├── moler.py
│   ├── molgpt.py
│   ├── regression_transformer.py
│   └── unimol.py
│
├── layers/                    # Neural network layers and building blocks
│   ├── __init__.py
│   ├── attention.py
│   ├── cache.py
│   ├── feed_forward.py
│   ├── layer_norm.py
│   ├── prediction.py
│   ├── rotary.py
│   └── transformer_layer.py
│
├── utils/                     # Utilities for models
│   ├── __init__.py
│   ├── containers.py          # ModelInput, ModelOutput data containers
│   └── state_dict_utils.py    # State dict cleaning/adaptation utilities
│
├── wrappers/                  # Wrappers for generation, encoding, etc.
│   ├── __init__.py
│   ├── encoding.py
│   └── generation.py
```

## Usage Examples

Import the main Hyformer model and registry:

```python
from hyformer.models import Hyformer, AutoModel

# Instantiate a model from config
generated_model = AutoModel.from_config(config)

# Load a pretrained model
model = AutoModel.from_pretrained('adamizdebski/hyformer-base')
```

Import a baseline model directly:

```python
from hyformer.models.baselines import GPT
```

## Extensibility & Best Practices
- **Add new models** by subclassing `PreTrainedModel` (in `core/base.py`) and registering with `AutoModel` using the `@AutoModel.register("model_type")` decorator.
- **Implement wrappers** for new generation or encoding strategies in the `wrappers/` submodule.
- **Reuse layers** from `layers/` for building new architectures.
- **Use utilities** from `utils/` for state dict handling and standardized model I/O.

## Contributing New Models or Wrappers
1. **Create your model** in `baselines/` or a new submodule, subclassing from the appropriate base class.
2. **Register your model** with `AutoModel` using the decorator:
   ```python
   from hyformer.models.auto import AutoModel

   @AutoModel.register("my_model_type")
   class MyModel(PreTrainedModel):
       ...
   ```
3. **(Optional) Add a wrapper** in `wrappers/` if your model needs custom generation or encoding logic.
