import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim


# Dummy model for demonstration
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.fc(x)


# Training function
def train():
    
    # Check if we are running in a distributed setting
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1 and torch.cuda.device_count() > 1

    if ddp:
        assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}')
        master_process = int(os.environ['RANK']) == 0
        seed_offset = int(os.environ['RANK'])
        if master_process: print(f"Running on {torch.cuda.device_count()} GPUs", flush=True)
    else:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        master_process = True
        seed_offset = 0
        print(f"Running on a single device: {device}.", flush=True)

    # Set the seed
    torch.manual_seed(13 + seed_offset)

    # Create the model and move it to the correct GPU
    model = DummyModel().to(device)
    ddp_model = DDP(model, device_ids=[device], find_unused_parameters=False) if ddp else model
    
    # Dummy optimizer and data
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    data = torch.randn(20, 10).to(device)
    target = torch.randn(20, 10).to(device)
    loss_fn = nn.MSELoss()
    
    # Simple training loop
    for epoch in range(50):  # Train for 5 epochs
        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        if master_process: print(f"Rank {os.environ.get('RANK', 0)}, Epoch {epoch}, Loss: {loss.item():.4f}", flush=True)

    # Cleanup
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


def main():
    train()


if __name__ == "__main__":
    main()
