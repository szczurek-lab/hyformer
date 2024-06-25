import torch

from typing import List
from guacamol.assess_distribution_learning import DistributionMatchingGenerator

class DefaultGuacamolModelWrapper(DistributionMatchingGenerator):
    def __init__(self, model, tokenizer, batch_size, temperature, top_k, device):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.model.eval()

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []
        model = self.model.to(self.device)
        while len(generated) < number_samples:
            samples = model.generate(self.tokenizer.cls_token_id,
                                        self.tokenizer.sep_token_id,
                                        self.tokenizer.pad_token_id,
                                        self.tokenizer.max_molecule_length,
                                        self.batch_size,
                                        self.temperature,
                                        self.top_k)
            generated.extend(self.tokenizer.decode(samples))
        return generated[:number_samples]
