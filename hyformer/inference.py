"""
Hyformer v1 inference — molecular (SMILES) sequences only.

Available checkpoints:
    - SzczurekLab/hyformer_molecules_8M: 8M parameters, 8 layers, embedding dim 256,
      pretrained on GuacaMol dataset [Brown et al.]
    - SzczurekLab/hyformer_molecules_50M: 50M parameters, 12 layers, embedding dim 512,
      pretrained on Uni-Mol dataset [Zhou et al.]

Pre-trained models expose three inference functions:
    - embed: CLS-token embeddings (len(sequences), embedding_dim)
    - predict: physicochemical property predictions in original descriptor units (len(sequences), num_properties)
    - compute_perplexity: sequence-level perplexity (len(sequences),)

References:
    Izdebski et al. "Synergistic Benefits of Joint Molecule Generation and Property Prediction"
    Brown et al. "GuacaMol: benchmarking models for de novo molecular design"
    Zhou et al. "Uni-mol: A universal 3d molecular representation learning framework"
"""
from typing import Literal, Optional

import numpy as np
import torch

from hyformer.models.auto import AutoModel
from hyformer.utils.data_loading import get_data_loader as _get_data_loader
from hyformer.utils.tokenizers.auto import AutoTokenizer


Checkpoint = Literal[
    "SzczurekLab/hyformer_molecules_8M",
    "SzczurekLab/hyformer_molecules_50M",
]


def embed(
    sequences: list[str],
    checkpoint: Checkpoint,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return CLS-token embeddings, shape (len(sequences), embedding_dim)."""
    model, tokenizer, device = _load(checkpoint, device=device)
    loader = _get_data_loader(
        dataset=sequences,
        tasks={"prediction": 1.0},
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
    )
    parts = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to_device(device)
            output = model(**batch, return_loss=False)
            parts.append(output["embeddings"][:, 0].cpu().numpy())
    return np.concatenate(parts, axis=0)


def predict(
    sequences: list[str],
    checkpoint: Checkpoint,
    batch_size: int = 32,
    device: Optional[str] = None,
    inverse_transform: bool = True,
) -> np.ndarray:
    """Return physicochemical property predictions, shape (len(sequences), num_properties).

    Args:
        inverse_transform: If True (default), invert the CDF normalization applied during
            training so predictions are in original physicochemical descriptor units.
            Set to False to obtain raw model outputs in [0, 1] space.
    """
    model, tokenizer, device = _load(checkpoint, device=device)
    assert hasattr(model, "physchem_head") and model.physchem_head is not None, (
        "This checkpoint does not have a physicochemical property prediction head. "
        "Use embed() or compute_perplexity() instead."
    )
    loader = _get_data_loader(
        dataset=sequences,
        tasks={"prediction": 1.0},
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
    )
    parts = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to_device(device)
            output = model(**batch, return_loss=False)
            parts.append(output["logits_physchem"].cpu().numpy())
    predictions = np.concatenate(parts, axis=0)
    if inverse_transform:
        from hyformer.utils.properties.smiles.molbert.featurizer import PhysChemFeaturizer
        scaler = PhysChemFeaturizer(normalise=True).scaler
        predictions = scaler.inverse_transform(predictions)
    return predictions


def compute_perplexity(
    sequences: list[str],
    checkpoint: Checkpoint,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return perplexity for each sequence, shape (len(sequences),)."""
    model, tokenizer, device = _load(checkpoint, device=device)
    loader = _get_data_loader(
        dataset=sequences,
        tasks={"generation": 1.0},
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
    )
    parts = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to_device(device)
            output = model(**batch, return_loss=False)
            parts.append(
                _sequence_perplexity(output["logits_generation"], batch["input_labels"]).cpu().numpy()
            )
    return np.concatenate(parts)


def _load(checkpoint: str, device: Optional[str] = None, local_dir: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_dir=local_dir)
    model = AutoModel.from_pretrained(checkpoint)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _sequence_perplexity(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    # shift by one: logits predict the next token
    logits = logits[:, :-1]
    targets = labels[:, 1:]
    mask = targets != ignore_index
    token_nll = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=ignore_index, reduction="none"
    )
    nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)
    return nll.exp()
