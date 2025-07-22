import random 
import torch

from typing import List, Optional, Union, Tuple

from hyformer.utils.tokenizers.base import TOKEN_DICT
from hyformer.utils.tokenizers.smiles_separate_task_token import SmilesTokenizerSeparateTaskToken
from hyformer.models.utils import ModelInput


class SmilesTokenizerSeparateTaskTokenFuture(SmilesTokenizerSeparateTaskToken):

    def __init__(
        self,
        path_to_vocabulary: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = TOKEN_DICT['ignore']
    ):

        super().__init__(
            path_to_vocabulary=path_to_vocabulary,
            max_molecule_length=max_molecule_length,
            mlm_probability=mlm_probability,
            ignore_index=ignore_index,
        )

    def __call__(self, x: Union[str, List[str], Tuple[str, torch.Tensor], List[Tuple[str, torch.Tensor]]], task: str) -> ModelInput:
        inputs = super().__call__(x, task)
        return self._trim(inputs)

    def _trim(self, batch):
        """Trim each sequence in a batch to a length sampled from a uniform distribution.
        """
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        task = batch['task']
        input_labels = batch.get('input_labels', None)
        properties = batch.get('properties', None)

        trimmed_input_ids = []
        trimmed_attention_mask = []
        trimmed_input_labels = [] if input_labels is not None else None

        for idx in range(input_ids.size(0)):
            sampled_length = random.randint(3, (input_ids[idx, :] == self.sep_token_id).nonzero()[0].item())
            trimmed_input_ids.append(torch.cat((input_ids[idx, :sampled_length], torch.tensor([self.sep_token_id], dtype=torch.long))))
            trimmed_attention_mask.append(torch.cat((attention_mask[idx, :sampled_length], torch.tensor([True], dtype=torch.bool))))
            trimmed_input_labels.append(torch.cat((input_labels[idx, :sampled_length], torch.tensor([self.sep_token_id], dtype=torch.long)))) if input_labels is not None else None
            
        # Pad sequences to the same length within the batch if necessary
        trimmed_input_ids = [torch.nn.functional.pad(seq, (0, input_ids.size(1) - len(seq)), value=self.pad_token_id) for seq in trimmed_input_ids]
        trimmed_attention_mask = [torch.nn.functional.pad(seq, (0, input_ids.size(1) - len(seq)), value=self.pad_token_id) for seq in trimmed_attention_mask]
        trimmed_input_labels = [torch.nn.functional.pad(seq, (0, input_labels.size(1) - len(seq)), value=self.pad_token_id) for seq in trimmed_input_labels] if input_labels is not None else None

        return ModelInput(
            input_ids=torch.stack(trimmed_input_ids),
            attention_mask=torch.stack(trimmed_attention_mask),
            task=task,
            input_labels=torch.stack(trimmed_input_ids) if input_labels is not None else None,
            properties=properties
        )
