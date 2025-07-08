from hyformer.configs.trainer import TrainerConfig
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Any

from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.tokenizers.base import BaseTokenizer

    
class DefaultEncoderWrapper:
    def __init__(self, model, tokenizer, batch_size, device):
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device

#     @torch.no_grad()
#     def encode(self, smiles: list[str]) -> np.ndarray:
#         self._model.eval()
#         model = self._model.to(self._device)
#         embeddings = []
#         for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
#             batch = smiles[i:i+self._batch_size]
#             batch_input = self._tokenizer(batch, task="prediction")
#             for k,v in batch_input.items():
#                 if isinstance(v, torch.Tensor):
#                     batch_input[k] = v.to(self._device)
#             output: ModelOutput = model(**batch_input, is_causal=False)
#             embeddings.append(output["global_embeddings"].cpu().numpy())
#         ret = np.concatenate(embeddings, axis=0)
#         return ret


class HyformerEncoderWrapper(DefaultEncoderWrapper):
    
    def __init__(self, model, tokenizer, batch_size, device):
        super().__init__(model, tokenizer, batch_size, device)
        
        from hyformer.configs.trainer import TrainerConfig
        from hyformer.trainers.trainer import Trainer

        _dummy_config = TrainerConfig(**{
                "batch_size": 0, "learning_rate": 0.0, "weight_decay": 0.0, "max_epochs": 0,
                "tasks": {'prediction': 1.0}, "compile": True, "enable_ddp": False, "dtype": "float32",
                "num_workers": 16, "beta1": 0.0, "beta2": 0.0, "gradient_accumulation_steps": 1,
                "grad_clip": 1.0, "decay_lr": False, "log_interval": 0, "save_interval": 0, "min_lr": 0.0
            })
        self._trainer = Trainer(config=_dummy_config, model=self._model, device=self._device, tokenizer=self._tokenizer)

    @torch.inference_mode()
    def encode(self, smiles: list[str]) -> np.ndarray:
        
        dataset = BaseDataset(data=smiles, target=None)
        _task = 'prediction'
        _tasks = {'prediction': 1.0}
        
        _cls_token_idx = 0
        _loader = self._trainer.create_loader(dataset=dataset, tasks=_tasks, batch_size=min(len(dataset), self._batch_size))
        embeddings = np.zeros((len(dataset), self._model.embedding_dim))
        for idx, batch in enumerate(tqdm(_loader, "Encoding samples")):
            batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = self._model(**batch, return_loss=False)
            _embeddings = output['embeddings'][:, _cls_token_idx].cpu().numpy()
            embeddings[idx*self._batch_size:(idx+1)*self._batch_size] = _embeddings
        return embeddings
    