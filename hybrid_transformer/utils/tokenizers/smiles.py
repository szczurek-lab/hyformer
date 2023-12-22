"""Source: https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html """

import torch

from typing import List, Tuple

from hybrid_transformer.utils.tokenizers.deepchem import DeepChemTokenizer
from hybrid_transformer.utils.objectives.guacamol.objective import get_objective

IGNORE_INDEX = -1


class SMILESTokenizer(DeepChemTokenizer):
    """ A mix between a BERT tokenizer and a data collator. """

    def __init__(
        self,
        path_to_vocab: str,
        max_molecule_length: int,
        mlm_probability: float = 0.15,
        ignore_index: int = IGNORE_INDEX
    ):

        super().__init__(
            vocab_file=path_to_vocab,
            cls_token='[GEN]',
            sep_token='[EOS]',
            mask_token="[MASK]",
            pad_token="[PAD]",
            unk_token = "[UNK]",
            additional_special_tokens=['[PRED]'])

        self.max_molecule_length = max_molecule_length
        self.mlm_probability = mlm_probability

        self.id_to_token = {key: item for item, key in self.vocab.items()}
        for id, special_token in enumerate(self.additional_special_tokens):
            self.id_to_token[self.additional_special_tokens_ids[id]] = self.additional_special_tokens[id]
        self.token_to_id = {item: key for item, key in self.vocab.items()}
        self.ignore_index = ignore_index

    def get_inputs(self, dataset, task, batch_size, device=None):

        sampled_idx = torch.randperm(len(dataset))[:batch_size]
        mlm_probability = self.mlm_probability

        if dataset.target_label is None:
            inputs = [dataset[idx] for idx in sampled_idx]
            target = None
        else:
            inputs = [dataset[idx][0] for idx in sampled_idx]
            target = torch.Tensor([dataset[idx][1] for idx in sampled_idx])
            mlm_probability = 0.0

        inputs = super().__call__(
            inputs, return_special_tokens_mask=True, padding='max_length', truncation=False,
            return_tensors='pt', max_length=128, return_token_type_ids=False)
        mask = inputs['attention_mask']
        eos_mask = inputs['input_ids'] == self.eos_token_id

        if task == 'lm':
            mlm_probability = 0.0
            inputs, labels = self.prepare_tokens(
                self.set_generation_task_token(inputs['input_ids']), inputs['special_tokens_mask'], mlm_probability)
        elif task == 'mlm':
            inputs, labels = self.prepare_tokens(
                self.set_prediction_task_token(inputs['input_ids']), inputs['special_tokens_mask'], mlm_probability)
        else:
            raise ValueError('Variable `task` must be either "seq2seq" or "mlm".')

        inputs = {'input_ids': inputs, 'attention_mask': mask, 'labels': labels, 'target': target, 'eos_mask': eos_mask}

        if device is not None:
            for key, value in inputs.items():
                inputs[key] = value.pin_memory().to(device, non_blocking=True)

        return inputs

    def encode(self, x: str or List[str]) -> torch.Tensor:
        if isinstance(x, str):
            x = list(x)
        encoded = torch.zeros((len(x), self.max_molecule_length))
        for idx, item in enumerate(encoded):
            encoded[idx] = super().encode(
                text=x[idx], padding='max_length', truncation=False,
                return_tensors='pt', max_length=self.max_molecule_length).flatten()
        return encoded

    def decode(self, x: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        smiles_data = []
        for _, row in enumerate(x):
            smiles = super().decode(row, skip_special_tokens=skip_special_tokens).replace(' ', '')
            smiles_data.append(smiles)
        return smiles_data

    @property
    def generate_token_id(self):
        return self.token_to_id['[GEN]']

    @property
    def predict_token_id(self):
        return self.token_to_id['[PRED]']

    @property
    def eos_token_id(self):
        return self.token_to_id['[EOS]']

    @property
    def mask_token_id(self):
        return self.token_to_id['[MASK]']

    @property
    def unk_token_id(self):
        return self.token_to_id['[UNK]']

    @property
    def pad_token_id(self):
        return self.token_to_id['[PAD]']

    def set_generation_task_token(self, x):
        x[:, 0] = self.generate_token_id
        return x

    def set_prediction_task_token(self, x):
        x[:, 0] = self.predict_token_id
        return x

    def prepare_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor,
                           mlm_probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 20% original.
        """

        labels = inputs.clone()
        indices_to_mask = torch.full(labels.shape, mlm_probability)

        if not special_tokens_mask.dtype == torch.bool:
            special_tokens_mask = special_tokens_mask.to(torch.bool)

        indices_to_mask.masked_fill_(special_tokens_mask, value=0.0)

        if mlm_probability > 0.0:

            # Get labels of masked tokens
            indices_to_mask = torch.bernoulli(indices_to_mask).bool()
            labels[~indices_to_mask] = self.ignore_index

            # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & indices_to_mask
            inputs[indices_replaced] = self.convert_tokens_to_ids(self.mask_token)
            # The rest of the time (20% of the time) we keep the masked input tokens unchanged

        else:
            indices_to_mask.masked_fill_(special_tokens_mask, value=1.0)
            indices_to_mask = indices_to_mask.bool()
            labels[indices_to_mask] = self.ignore_index

        return inputs, labels

    def is_valid_smiles(self, x: torch.Tensor) -> List[bool]:
        data_decoded = self.decode(x)
        is_valid_data = []
        for smiles in data_decoded:
            is_valid = is_valid(smiles)
            if is_valid is False:
                is_valid_data.append(False)
            else:
                is_valid_data.append(True)
        return is_valid_data

    def get_objective_values(self, data: torch.Tensor, task_id: str) -> torch.Tensor:
        return get_objective(self.decode(data), task_id).to(data.device)

    @classmethod
    def from_config(cls, config):
        return cls(path_to_vocab=config.path_to_vocab_file,  max_molecule_length=config.max_molecule_length)
