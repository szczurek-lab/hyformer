import os
import logging

from typing import Optional
from torch.distributed import init_process_group, destroy_process_group

console = logging.getLogger(__name__)
DDP_BACKEND = "nccl"


def init_ddp(enable_ddp: bool, rank: Optional[int] = None, world_size: Optional[int] = None) -> None:

    rank = rank if rank is not None else int(os.environ["SLURM_PROCID"])
    world_size = world_size if world_size is not None else int(os.environ["WORLD_SIZE"])

    if enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        init_process_group(backend=DDP_BACKEND, rank=rank, world_size=world_size)
        console.info("DDP initialized")
        
def end_ddp(enable_ddp: bool) -> None:
    if enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()
