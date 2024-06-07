import os
import logging

from torch.distributed import init_process_group, destroy_process_group

logger = logging.getLogger(__name__)


def init_ddp(enable_ddp: bool, backend: str) -> None:
    if enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        init_process_group(backend=backend)
        logger.info(f"DDP enabled with backend {backend}")


def end_ddp(enable_ddp: bool) -> None:
    if enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()
