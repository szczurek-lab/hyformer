import os
import logging

from torch.distributed import init_process_group, destroy_process_group

console = logging.getLogger(__name__)
DDP_BACKEND = "nccl"


def init_ddp(enable_ddp: bool) -> None:
    if enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        init_process_group(backend=DDP_BACKEND)
        console.info("DDP initialized")
        print("DDP initialized")
        
def end_ddp(enable_ddp: bool) -> None:
    if enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()
