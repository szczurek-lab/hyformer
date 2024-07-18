import torch

from typing import List
from base import SmilesEncoder, DistributionMatchingGenerator
from tqdm import tqdm

class DefaultGuacamolModelWrapper(DistributionMatchingGenerator):
    def __init__(self, model, tokenizer, batch_size, temperature, top_k, device):
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._temperature = temperature
        self._top_k = top_k

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []
        self._model.eval()
        model = self._model.to(self._device)
        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            samples = model.generate(self._tokenizer.cls_token_id,
                                        self._tokenizer.sep_token_id,
                                        self._tokenizer.pad_token_id,
                                        self._tokenizer.max_molecule_length,
                                        self._batch_size,
                                        self._temperature,
                                        self._top_k,
                                        self._device)
            generated.extend(self._tokenizer.decode(samples))
        return generated[:number_samples]

#class DefaultSmilesEncoderWrapper(SmilesEncoder):
#    def __init__(self, model, tokenizer, batch_size, device):
#        self._model = model
#        self._tokenizer = tokenizer
#        self._batch_size = batch_size
#        self._device = device
#
#    def encode(self, smiles: list[str]) -> torch.Tensor:
#        self._model.eval()
#        model = self._model.to(self._device)
#
#
#        
#        encodings = []
#        with self._model as model:
#            for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
#                enc = model.encode(smiles[i:i+self._batch_size])
#                encodings.extend(enc)
#        ret = torch.stack(encodings, dim=0)
#        return ret
#