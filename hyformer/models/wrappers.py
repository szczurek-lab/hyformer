import torch

from tqdm import tqdm
from typing import List, Any

from hyformer.utils.tokenizers.base import BaseTokenizer


class DefaultGeneratorWrapper:
    def __init__(
        self,
        model: Any,
        tokenizer: BaseTokenizer,
        batch_size: int,
        temperature: float, top_k: int,
        top_p: float,
        max_sequence_length: int,
        device: Any, compile: bool = False
        ) -> None:
        self._model = torch.compile(model) if compile else model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._max_sequence_length = max_sequence_length

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        samples = []
        self._model.eval()
        model = self._model.to(self._device)
        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            samples: list[str] = model.generate(self._tokenizer.cls_token_id,
                                        self._tokenizer.sep_token_id,
                                        self._tokenizer.pad_token_id,
                                        self._tokenizer.max_molecule_length,
                                        self._batch_size,
                                        self._temperature,
                                        self._top_k,
                                        self._top_p,
                                        self._device)
            samples.extend(self._tokenizer.decode(samples))
        return samples[:number_samples]


class HyformerGeneratorWrapper(DefaultGeneratorWrapper):
    
    @torch.inference_mode()
    def generate(self, number_samples: int) -> List[str]:
        
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
                num_tokens_to_generate=self._max_sequence_length - len(prefix_input_ids[0]),  # -2 for task and BOS tokens
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                use_cache=True
            )
            
            # Decode and add to samples list
            # for sequence in outputs:
            samples.extend(self._tokenizer.decode(outputs))
        
        self._model.train(_was_training)

        return samples[:number_samples]
    
    
# class DefaultSmilesEncoderWrapper(SmilesEncoder):
#     def __init__(self, model, tokenizer, batch_size, device):
#         self._model = model
#         self._tokenizer: SmilesTokenizer = tokenizer
#         self._batch_size = batch_size
#         self._device = device

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


# class HyformerSmilesEncoderWrapper(DefaultSmilesEncoderWrapper):

#     @torch.no_grad()
#     def encode(self, smiles: list[str]) -> np.ndarray:
#         self._model.eval()
#         model = self._model.to(self._device)
#         embeddings = np.zeros((len(smiles), model.embedding_dim))
#         for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
#             batch = smiles[i:i+self._batch_size]
#             model_input = self._tokenizer(batch, task="prediction")
#             model_input.to(self._device)
#             output: ModelOutput = model(**model_input)
#             embeddings[i:i+self._batch_size] = output.global_embeddings.cpu().numpy()
#         return embeddings
    