""" This script builds a vocabulary given a dataset and a tokenizer and optionally adds it to an existing vocabulary file. """

import os
import argparse
import re
import multiprocessing

from functools import partial
from tqdm.auto import tqdm

from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.experiments import save_strings_to_file


SMILES_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, default=None)
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--regex_pattern", type=str, required=False, default=SMILES_REGEX_PATTERN)
    parser.add_argument("--path_to_vocabulary_file", type=str, required=True)
    parser.add_argument("--num_processes", type=int, required=False, default=1, help="Number of processes to use for parallel processing. Default is 1.")
    args = parser.parse_args()
    return args

def _process_batch(batch_indices, dataset, regex):
        batch_tokens = set()
        for idx in batch_indices:
            smiles = dataset[idx]['data']
            tokens = regex.findall(smiles)
            batch_tokens.update(tokens)
        return batch_tokens

def main():

    args = parse_args()
    regex = re.compile(args.regex_pattern)    
    
    # If the vocabulary file exist, load existing tokens. Otherwise, create the file.
    if os.path.exists(args.path_to_vocabulary_file):
        with open(args.path_to_vocabulary_file, 'r') as f:
            vocabulary = set(line.strip() for line in f)
        print(f"Loaded existing vocabulary with {len(vocabulary)} tokens")
    else:
        vocabulary = set()
        print(f"Starting with empty vocabulary")
    
    _initial_vocabulary_size = len(vocabulary)
    
    dataset_config = DatasetConfig.from_config_filepath(args.path_to_dataset_config)
    _available_cpus = multiprocessing.cpu_count()
    _num_processes = min(args.num_processes, _available_cpus)
    print(f"Using {_num_processes} process{'es' if _num_processes > 1 else ''} out of {_available_cpus} available CPUs")

    for split in ['train', 'val', 'test']:
        try:
            print(f"Processing {split} split...")
            dataset = AutoDataset.from_config(dataset_config, split=split, root=args.data_dir)
            print(f"Loaded {len(dataset)} samples from {split} split")
            
            # Determine batch size - make batches smaller to reduce memory pressure
            batch_size = max(1, len(dataset) // (_num_processes * 20))
            print(f"Processing in batches of {batch_size} samples")
            
            # Create batches of indices
            all_indices = list(range(len(dataset)))
            batches = [all_indices[i:i+batch_size] for i in range(0, len(all_indices), batch_size)]
            
            # Process batches in parallel
            before_tokens = len(vocabulary)
            
            # Set up a single progress bar
            pbar = tqdm(
                total=len(batches),
                desc=f'Extracting tokens from {split}',
                position=0,
                leave=True,
                ncols=100,
                mininterval=0.5,
                smoothing=0.1,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            # Process batches and update the progress bar manually
            batch_results = []
            with multiprocessing.Pool(processes=_num_processes) as pool:
                process_func = partial(_process_batch, dataset=dataset, regex=regex)
                
                # Use imap to process batches
                for result in pool.imap(process_func, batches):
                    batch_results.append(result)
                    pbar.update(1)
            
            pbar.close()
            
            # Combine results
            for batch_tokens in batch_results:
                vocabulary.update(batch_tokens)
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
    
    # Update the final vocabulary file
    save_strings_to_file(sorted(list(vocabulary)), args.path_to_vocabulary_file)
    print(f'Final vocabulary saved to {args.path_to_vocabulary_file}')

    # Print the number of new tokens added
    new_tokens = len(vocabulary) - _initial_vocabulary_size
    print(f'Number of new tokens added: {new_tokens}')
    print(f'Total vocabulary size: {len(vocabulary)}')

if __name__ == "__main__":
    main()
