"""Preprocess SMILES strings.

This script preprocesses SMILES strings to use it with ML models. 

This script accepts accepts comma separated value files (.csv) as well as compressed (.csv.gz) files.

This file can also be imported as a module and contains the following
functions:
    * preprocess_smiles - preprocesses SMILES strings
"""

import os
import argparse

import numpy as np
import pandas as pd
import datamol as dm

from tqdm.auto import tqdm
from typing import List, Union, Optional
from multiprocessing import Pool, cpu_count

from sklearn.model_selection import train_test_split


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", type=str, required=True, help="Path to the input file containing SMILES strings.")
    parser.add_argument("--output_filepath_train", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--output_filepath_test", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--test_size", type=float, required=True, help="Size of the test set.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


def main(args):

    # Check if the file already exists, otherwise create a directory
    for output_filepath in [args.output_filepath_train, args.output_filepath_test]:
        assert not (os.path.exists(output_filepath) and os.path.isfile(output_filepath)), f"Output file already exists: {output_filepath}"
    
        dname = os.path.dirname(output_filepath)
        if not os.path.exists(dname):
            os.makedirs(dname, exist_ok=True)


    # Check if input file is a CSV or GZ file and load data
    input_filepath_extension = os.path.splitext(args.input_filepath)[1]
    assert input_filepath_extension in ['.csv', '.gz', '.npy'], "Input file format not supported."

    if input_filepath_extension == '.npy':
        smiles = np.load(args.input_filepath, allow_pickle=True)
    else:
        _pandas_compression_type = 'gzip' if input_filepath_extension == '.gz' else None
        smiles = pd.read_csv(args.input_filepath, compression=_pandas_compression_type, header=None)[0].values

    # Split data
    if args.test_size.is_integer():
        test_size = int(args.test_size)
        assert 0 < test_size and test_size < len(smiles), "Test size must smaller than the number of SMILES strings."
        test_size = float(test_size / len(smiles))
    else:
        test_size = args.test_size
        assert 0 < test_size and test_size < 1, "Test size must be an int or a float between 0 and 1."
    
        
    train_data, test_data = train_test_split(smiles, test_size=test_size, random_state=args.seed)
    np.save(args.output_filepath_train, train_data)
    np.save(args.output_filepath_test, test_data)

    print(f"Data saved to {args.output_filepath_train} and {args.output_filepath_test}")
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    