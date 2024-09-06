
from jointformer.models.base import BaseModel, SmilesEncoder
import os
import shutil
from tqdm import tqdm
import torch
import numpy as np
import unimol_tools


class UniMol(BaseModel, SmilesEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._dataset = None
        self._tokenizer = None
        self._batch_size = None
        self._temperature = None
        self._top_k = None
        self._device = None
    
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        return self

    def to_guacamole_generator(self, *args, **kwargs):
        raise NotImplementedError("UniMol is predictive only model")

    def encode(self, smiles: list[str]) -> np.ndarray:
        return np.stack(self._model.get_repr(smiles)["cls_repr"], axis=0)

    def load_pretrained(self, filename, *args, **kwargs):
        shutil.copy(filename, os.path.join(os.path.dirname(unimol_tools.__file__), "weights"))
        self._model = unimol_tools.UniMolRepr(data_type='molecule', remove_hs=False)
        
    @classmethod
    def from_config(cls, config):
        return cls()