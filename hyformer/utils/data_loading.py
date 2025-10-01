from torch.utils.data import DataLoader
from typing import Optional, Dict

from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.tokenizers.base import BaseTokenizer
from hyformer.utils.collator import DataCollator


def get_data_loader(
    dataset: Optional[BaseDataset],
    tasks: Dict[str, float],
    tokenizer: BaseTokenizer,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
    ):
    collator = DataCollator(tokenizer=tokenizer, tasks=tasks)
    return  DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                sampler=None,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=False
            )
