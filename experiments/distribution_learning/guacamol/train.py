import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record


local_rank = int(os.environ["LOCAL_RANK"])

# Need to have load_checkpoint, save checkpoint logic

def train():
  for batch in iter(dataset):
    train_step(batch)

    if should_checkpoint:
      save_checkpoint(checkpoint_path)


@record
def main():
    load_checkpoint(checkpoint_path)
    initialize()
    train()
    pass


if __name__ == "__main__":
  main()
