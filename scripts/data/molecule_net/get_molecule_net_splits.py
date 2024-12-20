import os
import random
import argparse

import pandas as pd
import numpy as np

from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split

from scripts.data.molecule_net.utils import load_molecule_net, read_file_info

RDLogger.logger().setLevel(RDLogger.CRITICAL) # Suppress RDKit warnings


def main(args):

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Detect and load tasks
    file_info = read_file_info(args.raw_data_dir)

    for filepath, task in file_info:
        
        print(f"Processing {task} from {filepath}...")
        smiles, targets = load_molecule_net(filepath, task)
        print(f"Loaded {len(smiles)} SMILES and {len(targets)} targets for {targets.shape[1]} tasks.")

        # create output directory 
        _output_dir_path = os.path.join(args.output_dir, task)
        if os.path.exists(_output_dir_path):
            continue
        os.makedirs(_output_dir_path, exist_ok=True)
        
        # split data into train / test
        if args.splitter == "hi":
            import lohi_splitter as lohi

            if len(smiles) <= 5000:
                train_idx, test_idx = lohi.hi_train_test_split(smiles, args.similarity_threshold, args.train_size, args.test_size)
            else:
                coarsening_threshold = 0.4
                max_mip_gap = 0.01
                train_idx, test_idx = lohi.hi_train_test_split(
                    smiles, args.similarity_threshold, args.train_size, args.test_size,
                    coarsening_threshold=coarsening_threshold, max_mip_gap=max_mip_gap)
            smiles_train, targets_train = [smiles[i] for i in train_idx], targets[train_idx]
            smiles_test, targets_test = [smiles[i] for i in test_idx], targets[test_idx]
        elif args.splitter == "random":
            smiles_train, smiles_test, targets_train, targets_test = train_test_split(smiles, targets, test_size=args.test_size, random_state=args.seed)
        elif args.splitter == "scaffold":
            smiles_train, smiles_test, targets_train, targets_test = lohi.scaffold_split(smiles, targets, args.train_size, args.test_size)
        else:
            raise ValueError(f"Invalid splitter: {args.splitter}")
        
        # split train into train / val
        # get random split for train / val
        smiles_train, smiles_val, targets_train, targets_val = train_test_split(smiles_train, targets_train, test_size=args.val_size, random_state=args.seed)

        print(f"Train: {len(smiles_train)} Val: {len(smiles_val)} Test: {len(smiles_test)}")

        np.savez(os.path.join(_output_dir_path, "train.npz"), sequence=smiles_train, properties=targets_train)
        np.savez(os.path.join(_output_dir_path, "val.npz"), sequence=smiles_val, properties=targets_val)
        np.savez(os.path.join(_output_dir_path, "test.npz"), sequence=smiles_test, properties=targets_test)

        print(f"Saved train, val, test data for {task} in {_output_dir_path}.")


def get_args():
    parser = argparse.ArgumentParser(description="Split MoleculeNet raw data into Hi split.")
    parser.add_argument("--raw-data-dir", type=str, required=True, help="Directory containing the raw data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the Hi split data.")
    parser.add_argument("--similarity-threshold", type=float, default=0.4, help="Similarity threshold for Hi split. Based on ECFP fingerprint.")
    parser.add_argument("--train-size", type=float, default=0.8, help="Train size for Hi split.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation size for Hi split.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test size for Hi split.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    parser.add_argument("--splitter", type=str, default="hi", help="Splitter to use for splitting data.")
    return parser.parse_args()     


if __name__ == "__main__":
    args = get_args()
    main(args=args)
