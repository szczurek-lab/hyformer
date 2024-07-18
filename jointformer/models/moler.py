from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from jointformer.models.base import BaseModel
from molecule_generation import load_model_from_directory
import os
from tqdm import tqdm

class Moler(BaseModel, DistributionMatchingGenerator):
    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._tokenizer = None
        self._batch_size = None
        self._temperature = None
        self._top_k = None
        self._device = None

    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._temperature = temperature
        self._top_k = top_k
        self._device = device
        return self

    def generate(self, number_samples: int):
        generated = []
        with self._model as model:
            for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
                samples = model.sample(self._batch_size)
                generated.extend(samples)
        return generated[:number_samples]


    def load_pretrained(self, filename, *args, **kwargs):
        dir = os.path.dirname(filename)
        self._model = load_model_from_directory(dir)
    
    @classmethod
    def from_config(cls, config):
        return cls()