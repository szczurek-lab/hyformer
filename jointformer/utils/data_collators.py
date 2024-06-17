import torch

from torch.distributions.categorical import Categorical


class DataCollator:

    def __init__(self, tokenizer, tasks):
        self.tokenizer = tokenizer
        self.tasks = tasks
        self._task_dist = Categorical(torch.Tensor(list(self.tasks.values())))

    def _sample_task(self):
        return list(self.tasks.keys())[self._task_dist.sample().item()]

    def __call__(self, items):
        task = self._sample_task()
        if isinstance(items[0], tuple):
            data = [item[0] for item in items]
            properties = [item[1] for item in items]
            inputs = self.tokenizer(data=data, properties=properties, task=task)
        else:
            inputs = self.tokenizer(data=items, task=task)
            inputs['properties'] = None
        return inputs
