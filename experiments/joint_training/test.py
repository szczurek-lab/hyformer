""" Test file that initializes a ddp run and counts the number of available GPUs.
"""

import os
import torch
import logging

from torch.distributed import init_process_group, destroy_process_group

DDP_BACKEND = "nccl"

logger = logging.getLogger(__name__)


def main():
    is_ddp_run = int(os.environ.get('RANK', -1)) != -1
    logger.info(f"DDP: {is_ddp_run}")

    if is_ddp_run:
        init_process_group(backend=DDP_BACKEND)

    if int(os.environ['RANK']) == 0:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {num_gpus}")

    if is_ddp_run:
        destroy_process_group()


if __name__ == "__main__":
    main()
