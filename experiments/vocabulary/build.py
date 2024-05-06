""" This script builds a vocabulary given a dataset and a tokenizer. """

import os
import argparse

from tqdm import tqdm

from jointformer.configs.task import TaskConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.utils.utils import save_strings_to_file

VOCABULARY_DIR = './data/vocabularies'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_task_config", type=str, required=True)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    task_config = TaskConfig.from_pretrained(args.path_to_task_config)

    dataset = AutoDataset.from_config(task_config, split='all')
    tokenizer = AutoTokenizer.from_config(task_config)

    vocabulary = set()
    for idx, x in enumerate(tqdm(dataset, desc='Extracting all tokens')):
        if hasattr(tokenizer, 'basic_tokenizer'):
            tokens = tokenizer.basic_tokenizer.tokenize(x)
        else:
            raise NotImplementedError
        vocabulary.update(set(tokens))

    out_dir = os.path.join(VOCABULARY_DIR, f'{task_config.dataset_name.lower()}.txt')
    save_strings_to_file(list(vocabulary), out_dir)
    print(f'Vocabulary saved to {out_dir}')


if __name__ == "__main__":
    main()
