import torch
import numpy as np
from tqdm import tqdm
from typing import List, Any, Optional

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
    def encode(self, sequences: list[str]) -> np.ndarray:

        self._model.eval()
        model = self._model.to(self._device)

        _cls_token_idx = 0        
        self._dataloader = create_dataloader(
            dataset=sequences,
            tasks={"prediction": 1.0},
            tokenizer=self._tokenizer,
            batch_size=min(len(sequences), self._batch_size),
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


###
# Generation
###

_MAX_SEQUENCE_LENGTH = 512

class BaseGeneratorWrapper:
    def __init__(
        self,
        model: Any,
        tokenizer: BaseTokenizer,
        batch_size: int,
        device: Any
        ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._max_sequence_length = _MAX_SEQUENCE_LENGTH

    @torch.inference_mode()
    def generate(
        self,
        number_samples: int,
        seed: int = 1337,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        ) -> List[str]:
        pass


class HyformerGeneratorWrapper(BaseGeneratorWrapper):
    
    def __init__(
        self,
        model: Any,
        tokenizer: BaseTokenizer,
        batch_size: int,
        device: Any
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device
        )
        
    @torch.inference_mode()
    def generate(
        self,
        number_samples: int,
        seed: int = 1337,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        ) -> List[str]:
        
        # set model to evaluation mode
        _was_training = self._model.training
        self._model.eval()
        model = self._model.to(self._device)
        
        # initialize samples
        samples = []
        
        # Create initial input for generation (task token + BOS token)
        prefix_input_ids = torch.tensor(
            [[self._tokenizer.task_token_id('lm'), self._tokenizer.bos_token_id]] * self._batch_size,
            dtype=torch.long,
            device=self._device
        )

        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            outputs = model.generate(
                prefix_input_ids=prefix_input_ids,
                num_tokens_to_generate=self._max_sequence_length - len(prefix_input_ids[0]), 
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=None,
                use_cache=False,
                seed=seed
            )
            samples.extend(self._tokenizer.decode(outputs))
        
        self._model.train(_was_training)

        return samples[:number_samples]
    
###
# Prediction
###

class BasePredictorWrapper:
    def __init__(self, model: Any, tokenizer: BaseTokenizer, batch_size: int, device: Any):
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
    
    @torch.inference_mode()
    def predict(self, sequences: list[str]) -> np.ndarray:
        raise NotImplementedError

class HyformerPredictorWrapper(BasePredictorWrapper):
    
    def __init__(self, model: Any, tokenizer: BaseTokenizer, batch_size: int, device: Any):
        super().__init__(model, tokenizer, batch_size, device)
        
    @torch.inference_mode()
    def predict(self, sequences: list[str]) -> np.ndarray:
        self._model.eval()
        model = self._model.to(self._device)
 
        self._dataloader = create_dataloader(
            dataset=sequences,
            tasks={"prediction": 1.0},
            tokenizer=self._tokenizer,
            batch_size=min(len(sequences), self._batch_size),
            shuffle=False,
        )

        predictions_list = []
        for batch in tqdm(self._dataloader, "Predicting samples"):
            batch = batch.to_device(self._device)
            output = model.predict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            predictions_list.append(output.detach().cpu().numpy())
        ret = np.concatenate(predictions_list, axis=0)
        return ret
