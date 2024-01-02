import torch

from typing import List
from guacamol.assess_distribution_learning import DistributionMatchingGenerator


class GuacamolModelWrapper(DistributionMatchingGenerator):

    def __init__(self, model, tokenizer, batch_size, device):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self._eval()

    def _eval(self):
        if self.model.training:
            self.model.eval()

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []

        while len(generated) < number_samples:
            idx = torch.ones(size=(self.batch_size, 1), device=self.device) * self.tokenizer.generate_token_id
            idx = idx.long()
            samples = self.model.generate(idx=idx, max_new_tokens=self.tokenizer.max_molecule_length)
            generated += self.tokenizer.decode(samples)
        return generated[:number_samples]
