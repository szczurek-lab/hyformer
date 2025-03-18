""" This script builds a vocabulary given a dataset and a tokenizer. """

import os
import argparse
import torch

from tqdm import tqdm

from hyformer.configs.dataset import DatasetConfig

from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.tokenizers.smiles.regex import RegexSmilesTokenizer
from hyformer.utils.runtime import save_strings_to_file

VOCABULARY_DIR = './data/vocabularies'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_task_config", type=str, required=True)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)

    dataset = AutoDataset.from_config(dataset_config, split='all')
    print(f"Number of SMILES strings: {len(dataset)}")
    tokenizer = RegexSmilesTokenizer()

    vocabulary = set()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    for batch in tqdm(dataloader, desc='Extracting all tokens'):
        # Handle different batch formats
        if isinstance(batch, dict):
            # New dictionary format
            smiles_batch = batch['data']
        elif isinstance(batch, tuple):
            # Old tuple format (data, target)
            smiles_batch = batch[0]
        else:
            # Direct data format
            smiles_batch = batch
        
        # Ensure smiles_batch is iterable
        if not isinstance(smiles_batch, (list, tuple)):
            smiles_batch = [smiles_batch]
            
        for smiles in smiles_batch:
            tokens = tokenizer.tokenize(smiles)
            vocabulary.update(set(tokens))

    out_dir = os.path.join(VOCABULARY_DIR, f'{task_config.dataset_type.lower()}.txt')
    save_strings_to_file(list(vocabulary), out_dir)
    print(f'Vocabulary saved to {out_dir}')


if __name__ == "__main__":
    main()
