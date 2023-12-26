from transformers import PretrainedConfig, PreTrainedTokenizer
from typing import List

from hybrid_transformer.utils.datasets.guacamol import GuacamolSMILESDataset

import importlib


class TaskConfig(PretrainedConfig):
    task_type = "distribution_learning"

    def __init__(
        self,
        out_dir: str = None,
        dataset_name: str = "guacamol",
        molecular_representation: str = 'SMILES',
        augment_molecular_representation: bool = True,
        augmentation_prob: float = 0.8,
        subset_dataset: bool or int = False,
        tokenizer: str = 'SMILESTokenizer',
        split: str = 'train',
        target_label: str = None,
        validate: bool = True,
        **kwargs,
    ):
        if dataset_name not in ["guacamol",]:
            raise ValueError(f"`dataset` must be 'guacamol', got {dataset_name}.")
        if molecular_representation not in ["SMILES"]:
            raise ValueError(f"`molecular_representation` must be 'SMILES, got {molecular_representation}.")

        # Output
        self.out_dir = out_dir

        # Data config
        self.dataset_name = dataset_name
        self.molecular_representation = molecular_representation
        self.tokenizer = tokenizer

        # Dataset config
        self.split = split
        self.target_label = target_label
        self.augment_molecular_representation = augment_molecular_representation
        self.augmentation_prob = augmentation_prob
        self.subset_dataset = subset_dataset
        self.validate = validate

        # Tokenizer
        self.path_to_vocab_file = './vocabularies/smiles.txt'
        self.max_molecule_length = 128

        # Attention mask
        self.use_pad_token_attention_mask = False

        super().__init__(**kwargs)

    def save(self, save_directory: str) -> None:
        super().save_pretrained(save_directory=save_directory)

    def load(self, config_path: str) -> 'PretrainedConfig':
        return super().from_pretrained(pretrained_model_name_or_path=config_path)
