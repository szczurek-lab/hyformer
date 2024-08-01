import torch

from torch.distributions.categorical import Categorical
from jointformer.models.utils import ModelInput

class DataCollator:

    def __init__(self, tokenizer, tasks):
        self.tokenizer = tokenizer
        self.tasks = tasks
        self._task_dist = Categorical(torch.Tensor(list(self.tasks.values())))

    def _sample_task(self):
        return list(self.tasks.keys())[self._task_dist.sample().item()]

    def __call__(self, batch):
        task = self._sample_task()
        return self.tokenizer(batch, task=task)
        