"""Hyformer package."""

__version__ = "1.0.0"

import warnings
warnings.warn(
    "Hyformer v1: designed for molecular (SMILES) sequences."
    "For other sequence types see Hyformer 2.0 (https://github.com/szczurek-lab/hyformer/tree/hyformer-2.0).",
    UserWarning,
    stacklevel=2,
)


from hyformer.models.auto import AutoModel
from hyformer.models.hyformer import Hyformer
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.inference import embed, predict, compute_perplexity

__all__ = ["AutoModel", "Hyformer", "AutoTokenizer", "__version__", "embed", "predict", "compute_perplexity"]
