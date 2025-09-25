"""
Utility functions for data loading.

This module provides utility functions that are used across the data loading
infrastructure but don't belong to any specific class.
"""

import os
import torch
from typing import Dict, Optional, List, Union, Any, Tuple

from torch.utils.data import DataLoader

from hyformer.utils.data.datasets.base import BaseDataset
from hyformer.tokenizers.base import BaseTokenizer
from hyformer.utils.reproducibility import seed_worker, get_global_seed


SUPPORTED_TASKS = ["lm", "prediction", "mlm"]
_PAD_TO_MULTIPLE_OF = 64
_LM_PREFIX_LENGTH = 2  # e.g., [TASK], [BOS]


def _pad_sequences(
    sequences: List[Union[torch.Tensor, List[int]]], max_length: int, pad_value: Any
) -> torch.Tensor:
    if not isinstance(sequences[0], torch.Tensor):
        sequences = [torch.tensor(seq) for seq in sequences]
    batch_size = len(sequences)
    padded = torch.full(
        (batch_size, max_length),
        pad_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length].clone()
    return padded


def _get_special_tokens_mask(
    tokenizer: BaseTokenizer, input_ids: torch.Tensor
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for tok_id in tokenizer.all_special_ids:
        if tok_id is not None:
            mask |= input_ids == tok_id
    if input_ids.dim() > 1:
        mask[:, 0] = True
        mask[:, 1] = True
    else:
        mask[0] = True
        mask[1] = True
    return mask


def _mask_tokens(
    tokenizer: BaseTokenizer,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = inputs.clone()
    special_mask = _get_special_tokens_mask(tokenizer, inputs)
    probability_matrix = torch.full(labels.shape, prob, device=labels.device)
    probability_matrix.masked_fill_(special_mask, value=0.0)
    probability_matrix.masked_fill_(attention_mask.eq(0), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    to_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool()
        & masked_indices
    )
    inputs[to_mask] = tokenizer.mask_token_id
    to_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool()
        & masked_indices
        & ~to_mask
    )
    random_tokens = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long, device=labels.device
    )
    inputs[to_random] = random_tokens[to_random]
    return inputs, labels


def create_dataloader(
    dataset: Optional[BaseDataset],
    tasks: Dict[str, float],
    tokenizer: BaseTokenizer,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    max_length: Optional[int] = None,
    mask_probability: float = 0.15,
    worker_seed: Optional[int] = None,
    **kwargs,
) -> Optional[torch.utils.data.DataLoader]:
    """Create a DataLoader for sequence data with a minimal built-in collator.

    - Deterministic seeding: uses `worker_seed` or the global seed to seed a generator
      and per-worker seeds via `worker_init_fn`.
    - Supported tasks: 'lm', 'prediction', 'mlm'
    """
    if dataset is None:
        return None

    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task: {task}. Supported tasks are {SUPPORTED_TASKS}."
            )

    # Normalize task probabilities once
    total = sum(tasks.values())
    task_items = list(tasks.items())
    task_probs = [w / total for _, w in task_items]

    def _sample_task() -> str:
        idx = torch.multinomial(torch.tensor(task_probs, dtype=torch.float), 1).item()
        return task_items[idx][0]

    pad_to = pad_to_multiple_of or _PAD_TO_MULTIPLE_OF
    max_len = max_length or 512

    def collate_fn(batch: List[Dict[str, Any]]):
        task = _sample_task()
        # Support both dataset items as dicts (with 'data'/'target') and raw strings
        is_mapping = isinstance(batch[0], dict)
        texts = [ex["data"] for ex in batch] if is_mapping else batch
        tokenized = tokenizer(texts, task=task)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        batch_max_len = min(max_len, max(len(ids) for ids in input_ids))
        if task == "lm":
            batch_max_len += 1
        if pad_to is not None:
            batch_max_len = ((batch_max_len + pad_to - 1) // pad_to) * pad_to

        padded_input_ids = _pad_sequences(
            input_ids, batch_max_len, pad_value=tokenizer.pad_token_id
        )
        padded_attn = _pad_sequences(attention_mask, batch_max_len, pad_value=False)

        input_labels = None
        model_input_ids = padded_input_ids
        target = None

        if task == "lm":
            input_labels = padded_input_ids.clone()
            input_labels[:, :_LM_PREFIX_LENGTH] = -100
            input_labels[input_labels == tokenizer.pad_token_id] = -100
        elif task == "mlm":
            model_input_ids, input_labels = _mask_tokens(
                tokenizer, padded_input_ids.clone(), padded_attn, mask_probability
            )
        elif task == "prediction":
            if is_mapping:
                targets = [ex["target"] for ex in batch]
                if not all(t is None for t in targets):
                    import numpy as _np

                    target = torch.from_numpy(_np.stack(targets)).float()

        from hyformer.models.utils import ModelInput  # local import to avoid circulars

        return ModelInput(
            input_ids=model_input_ids,
            attention_mask=padded_attn,
            task=task,
            input_labels=input_labels,
            target=target,
        )

    # Resolve workers and seeding
    resolved_workers = (
        int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
        if num_workers is None
        else num_workers
    )
    g = torch.Generator()
    g.manual_seed(worker_seed if worker_seed is not None else get_global_seed())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=resolved_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=resolved_workers > 0,
        prefetch_factor=2 if resolved_workers > 0 else None,
        worker_init_fn=seed_worker if resolved_workers > 0 else None,
        generator=g,
        **kwargs,
    )
    return loader
