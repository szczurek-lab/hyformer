import torch
import numpy as np
from tqdm import tqdm
from typing import List, Any

from hyformer.utils.data.datasets.base import BaseDataset
from hyformer.tokenizers.base import BaseTokenizer
from hyformer.utils.data.utils import create_dataloader
from hyformer.models.utils.containers import ModelInput, ModelOutput

###
# Featurization
###
    
class BaseEncoderWrapper:
    def __init__(self, model: Any, tokenizer: BaseTokenizer, batch_size: int, device: Any):
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._dataloader = None
        
    @torch.inference_mode()
    def encode(self, smiles: list[str]) -> np.ndarray:
        raise NotImplementedError
    

class HyformerEncoderWrapper(BaseEncoderWrapper):
    
    def __init__(self, model: Any, tokenizer: BaseTokenizer, batch_size: int, device: Any):
        super().__init__(model, tokenizer, batch_size, device)

    @torch.inference_mode()
    def encode(self, smiles: list[str]) -> np.ndarray:

        self._model.eval()
        model = self._model.to(self._device)

        _cls_token_idx = 0        
        self._dataloader = create_dataloader(
            dataset=smiles,
            tasks={"prediction": 1.0},
            tokenizer=self._tokenizer,
            batch_size=min(len(smiles), self._batch_size),
            shuffle=False,
        )
        
        embeddings_list = []
        for batch in tqdm(self._dataloader, "Encoding samples"):
            batch = batch.to_device(self._device)
            output: ModelOutput = model(**batch, return_loss=False)
            batch_embeddings = output["embeddings"][:, _cls_token_idx].detach().cpu().numpy()
            embeddings_list.append(batch_embeddings)
        ret = np.concatenate(embeddings_list, axis=0)
        return ret


# class DefaultGeneratorWrapper:
#     def __init__(
#         self,
#         model: Any,
#         tokenizer: BaseTokenizer,
#         batch_size: int,
#         temperature: float, top_k: int,
#         top_p: float,
#         max_sequence_length: int,
#         device: Any,
#         compile: bool = False,
#         use_cache: bool = False
#         ) -> None:
#         self._model = torch.compile(model) if compile else model
#         self._tokenizer = tokenizer
#         self._batch_size = batch_size
#         self._device = device
#         self._temperature = temperature
#         self._top_k = top_k
#         self._top_p = top_p
#         self._max_sequence_length = max_sequence_length
#         self._use_cache = use_cache
        
#     @torch.no_grad()
#     def generate(self, number_samples: int) -> List[str]:
#         pass


# class HyformerGeneratorWrapper(DefaultGeneratorWrapper):
    
#     def load_pretrained(self, filepath: str, discard_prediction_head: bool = False):
#         checkpoint = torch.load(filepath, map_location=self._device)["model"]
#         self._model.load_pretrained(state_dict=checkpoint, discard_prediction_head=discard_prediction_head)
        
#     @torch.inference_mode()
#     def generate(self, number_samples: int, temperature: float = None, top_k: int = None, top_p: float = None) -> List[str]:
        
#         # set model to evaluation mode
#         _was_training = self._model.training
#         self._model.eval()
#         model = self._model.to(self._device)
        
#         temperature = temperature if temperature is not None else self._temperature
#         top_k = top_k if top_k is not None else self._top_k
#         top_p = top_p if top_p is not None else self._top_p
        
#         # initialize samples
#         samples = []
        
#         # Create initial input for generation (task token + BOS token)
#         prefix_input_ids = torch.tensor(
#             [[self._tokenizer.task_token_id('lm'), self._tokenizer.bos_token_id]] * self._batch_size,
#             dtype=torch.long,
#             device=self._device
#         )

#         for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
#             outputs = model.generate(
#                 prefix_input_ids=prefix_input_ids,
#                 num_tokens_to_generate=self._max_sequence_length - len(prefix_input_ids[0]), 
#                 eos_token_id=self._tokenizer.eos_token_id,
#                 pad_token_id=self._tokenizer.pad_token_id,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p,
#                 use_cache=self._use_cache
#             )
#             samples.extend(self._tokenizer.decode(outputs))
        
#         self._model.train(_was_training)

#         return samples[:number_samples]
    