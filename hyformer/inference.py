from typing import Optional

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
