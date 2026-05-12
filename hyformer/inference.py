"""
Hyformer v2 inference — peptide sequences.

Available checkpoints:
    - SzczurekLab/hyformer_peptides_34M: 34M parameters, pretrained on 3.5M general-purpose
      and antimicrobial peptides.
    - SzczurekLab/hyformer_peptides_34M_MIC: hyformer_peptides_34M jointly fine-tuned on
      minimal inhibitory concentration (MIC) values against E. coli bacteria.

Pre-trained models expose three inference functions:
    - embed: CLS-token embeddings (len(sequences), embedding_dim)
    - predict: property predictions (len(sequences), num_properties)
    - compute_perplexity: sequence-level perplexity (len(sequences),)

References:
    Izdebski et al. "Synergistic Benefits of Joint Molecule Generation and Property Prediction"
"""
from typing import Literal, Optional

_CHECKPOINT = Literal[
    "SzczurekLab/hyformer_peptides_34M",
    "SzczurekLab/hyformer_peptides_34M_MIC",
]
_CHECKPOINT_MIC = Literal["SzczurekLab/hyformer_peptides_34M_MIC"]

import numpy as np
import torch

from hyformer.models.auto import AutoModel
from hyformer.tokenizers.auto import AutoTokenizer


def embed(
    sequences: list[str],
    checkpoint: str,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return CLS-token embeddings, shape (len(sequences), embedding_dim)."""
    model, tokenizer, device = _load(checkpoint, device=device)
    return model.to_encoder(tokenizer, batch_size, device).encode(sequences)


def predict(
    sequences: list[str],
    checkpoint: str,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return property predictions, shape (len(sequences), num_properties).

    Args:
        checkpoint: Must be a checkpoint fine-tuned for property prediction
            (e.g. ``SzczurekLab/hyformer_peptides_34M_MIC``). Base generative
            checkpoints do not have a prediction head — use :func:`embed` or
            :func:`compute_perplexity` instead.
    """
    model, tokenizer, device = _load(checkpoint, device=device)
    assert hasattr(model, "prediction_head") and model.prediction_head is not None, (
        "This checkpoint does not have a prediction head. "
        "Use embed() or compute_perplexity() instead."
    )
    return model.to_predictor(tokenizer, batch_size, device).predict(sequences)


def compute_perplexity(
    sequences: list[str],
    checkpoint: str,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return perplexity for each sequence, shape (len(sequences),)."""
    model, tokenizer, device = _load(checkpoint, device=device)
    from hyformer.utils.data.utils import create_dataloader
    loader = create_dataloader(
        dataset=sequences,
        tasks={"lm": 1.0},
        tokenizer=tokenizer,
        batch_size=min(len(sequences), batch_size),
        shuffle=False,
    )
    parts = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to_device(device)
            output = model(**batch, return_loss=False)
            parts.append(
                _sequence_perplexity(output["logits"], batch["input_labels"]).cpu().numpy()
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
