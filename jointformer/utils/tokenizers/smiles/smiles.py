""" Implements a SMILES tokenizer for the Jointformer model. """

import torch

from typing import List, Tuple, Any, Optional, Union

from guacamol.utils.chemistry import is_valid

from jointformer.utils.tokenizers.smiles.deepchem import DeepChemSmilesTokenizer

IGNORE_INDEX = -100  # same as in transformers library
PREDICTION_TOKEN = '[PRED]'
RECONSTRUCTION_TOKEN = '[REC]'
GENERATION_TOKEN = '[BOS]'


class SmilesTokenizer(DeepChemSmilesTokenizer):
    """ A SMILES tokenizer and a data collator. """

    def __init__(
        self,
        path_to_vocabulary: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = IGNORE_INDEX,
        set_separate_task_tokens: Optional[bool] = True,
    ):

        super().__init__(
            path_to_vocabulary=path_to_vocabulary,
            cls_token='[GEN]',
            sep_token='[EOS]',
            mask_token="[MASK]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            additional_special_tokens=[PREDICTION_TOKEN, RECONSTRUCTION_TOKEN, GENERATION_TOKEN])

        self.max_molecule_length = max_molecule_length
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index
        self.set_separate_task_tokens = set_separate_task_tokens
        self.id_to_token = None
        self.token_to_id = None
        self._post_init()

    def _post_init(self):

        # if self.set_separate_task_tokens:
        #     self.add_tokens([PREDICTION_TOKEN, RECONSTRUCTION_TOKEN, GENERATION_TOKEN], special_tokens=True)

        self.id_to_token = {key: item for item, key in self.vocab.items()}
        for id, special_token in enumerate(self.additional_special_tokens):
            self.id_to_token[self.additional_special_tokens_ids[id]] = self.additional_special_tokens[id]
        self.token_to_id = {item: key for item, key in self.vocab.items()}

    def __call__(self, examples: str or List[str], task: str = 'lm') -> dict:
        """ Tokenize a SMILES string or a list of SMILES strings.

        Parameters
        ----------
        examples : str or List[str]
            A SMILES string or a list of SMILES strings.
        task : str
            The task to perform. It can be either 'lm', 'ae', 'prediction' or 'mlm'.

        Returns
        -------
        dict
            A dictionary containing the tokenized SMILES strings. `Input_ids` contains the tokenized SMILES strings,
            `attention_mask` contains the attention mask, with padding tokens masked, `special_tokens_mask`,
             with SMILES tokens masked and 'labels' for computing the LM loss.
        """

        batch = super().__call__(
            examples, return_special_tokens_mask=True, padding='max_length', truncation=True,
            return_tensors='pt', max_length=self.max_molecule_length, return_token_type_ids=False)
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if task == 'lm' or task == 'ae':
            labels = batch["input_ids"].clone()
            if self.pad_token_id is not None:
                labels[labels == self.pad_token_id] = IGNORE_INDEX
            batch["labels"] = labels
            # todo: set the token to mlm

        elif task == 'mlm':
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask)
            # todo: set the token to mlm

        elif task == 'prediction':
            pass
            # todo: set the token to mlm

        else:
            raise ValueError('Variable `task` must be either `lm`, `ae`, `prediction` or `mlm`.')

        return batch

    def set_separate_task_tokens(self, x: dict, task: str) -> dict:
        if task == 'prediction':
            task_token = self.predict_token_id
        elif task == 'lm':
            task_token = self.generate_token_id
        elif task == 'ae':
            task_token = self.reconstruct_token_id
        elif task == 'mlm':
            task_token = self.reconstruct_token_id
        else:
            raise ValueError('Variable `task` must be either `lm`, `ae`, `prediction` or `mlm`.')
        return self.set_task_token(x, task_token)

    def set_task_token(self, x: dict, task_token: int) -> dict:
        for input in ['input_ids', 'labels']:
            if input in x:
                x[input][:, 0] = task_token
                x[input] = x['input_ids'].contiguous()
        return x

    def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Reimplements preparing masked token inputs / labels for masked language modeling: 80% MASK, 20% original.

        No random token replacement is performed, as this would correspond to invalid SMILES strings.

        Source: https://github.com/huggingface/transformers/blob/745bbfe4bb2b61491dedd56e1e8ee4af8ef1a9ec/src/transformers/data/data_collator.py#L782
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = IGNORE_INDEX  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.convert_tokens_to_ids(self.mask_token)

        # We do not replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (20% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def decode(self, x: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        smiles_data = []
        for _, row in enumerate(x):
            smiles = super().decode(row, skip_special_tokens=skip_special_tokens).replace(' ', '')
            smiles_data.append(smiles)
        return smiles_data

    def set_generation_task_token(self, x):
        x[:, 0] = self.generate_token_id
        return x

    def set_prediction_task_token(self, x):
        x[:, 0] = self.predict_token_id
        return x

    def is_valid_smiles(self, x: Union[torch.Tensor, List[str], str]) -> List[bool]:
        if isinstance(x, str):
            return is_valid(x)
        elif isinstance(x, list):
            is_valid_data = []
            for smiles in x:
                valid = is_valid(smiles)
                if valid is False:
                    is_valid_data.append(False)
                else:
                    is_valid_data.append(True)
            return is_valid_data
        elif isinstance(x, torch.Tensor):
            return self.is_valid_smiles(self.decode(x))
        else:
            raise ValueError('Input must be a tensor, a list of strings or a string.')

    @classmethod
    def from_config(cls, config):
        return cls(path_to_vocabulary=config.path_to_vocabulary,  max_molecule_length=config.max_molecule_length)
