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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", type=str, required=True, help="Path to the input file containing SMILES strings.")
    parser.add_argument("--output_filepath", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--max_smiles_length", type=int, required=True, help="Maximum length of the SMILES strings.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers for parallel processing.")
    return parser


def _preprocess(smiles: str, max_length: int) -> Union[None, str]:

    _DISABLE_RDKIT_LOGS = True
    with dm.without_rdkit_log(enable=_DISABLE_RDKIT_LOGS):
        
        try:
            mol = dm.to_mol(smiles, ordered=True)
            mol = dm.fix_mol(mol)
            mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
            mol = dm.standardize_mol(
                mol,
                disconnect_metals=False,
                normalize=True,
                reionize=True,
                uncharge=False,
                stereo=True,
            )
            standardized_smile = dm.standardize_smiles(dm.to_smiles(mol))
            
            ###
            # Alternatively: https://github.com/valence-labs/mood-experiments/blob/main/mood/preprocessing.py#L24
            #
            # mol = dm.to_mol(smile, ordered=True, sanitize=False)
            # mol = dm.sanitize_mol(mol)
            # mol = dm.standardize_mol(mol)
            # standardized_smile = dm.to_smiles(mol)
            ###

            standardized_smile = smiles
            assert len(standardized_smile) <= max_length

        except:
            return None
        
    return standardized_smile


def _worker(args):
    # Wrapper to unpack arguments for multiprocessing
    return _preprocess(*args)


def preprocess(smiles: List[str], max_length: int, n_jobs: Optional[int] = 1, disable_logs: Optional[bool] = False, properties=None) -> List[str]:
    """
    Parallelized preprocessing of a list of SMILES strings.
    """
    
    assert len(smiles) == len(properties), "Length of SMILES and properties should be the same."
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    with Pool(n_jobs) as pool:
        args = [(smile, max_length) for smile in smiles]
        results = list(
            tqdm(
                pool.imap(_worker, args),
                total=len(smiles),
                desc="Preprocessing SMILES strings",
                disable=disable_logs,
            )
        )

    if properties is not None:
        return [res for res, prop in zip(results, properties) if res is not None], [prop for res, prop in zip(results, properties) if res is not None]
    return [res for res in results if res is not None]
    
    
def main(args):

    # Check if the file already exists, otherwise create a directory
    assert not (os.path.exists(args.output_filepath) and os.path.isfile(args.output_filepath)), f"Output file already exists: {args.output_filepath}"
    
    dname = os.path.dirname(args.output_filepath)
    if not os.path.exists(dname):
        os.makedirs(dname, exist_ok=True)


    # Check if input file is a CSV or GZ file and load data
    input_filepath_extension = os.path.splitext(args.input_filepath)[1]
    assert input_filepath_extension in ['.csv', '.gz'], "Input file format not supported."

    _pandas_compression_type = 'gzip' if input_filepath_extension == '.gz' else None
    smiles = pd.read_csv(args.input_filepath, compression=_pandas_compression_type, header=None)[0].values

    # Preprocess data
    smiles_processed, properties_processed = preprocess(smiles, max_length=args.max_smiles_length, n_jobs=args.n_workers, properties=np.load(args.input_properties_path) if args.input_properties_path else None)

    # Save data
    output_filepath_extension = os.path.splitext(args.output_filepath)[1]
    assert output_filepath_extension in ['.csv', '.gz', '.npy'], "Input file format not supported."
    
    if output_filepath_extension == '.npy':
        np.save(args.output_filepath, smiles_processed)
    else:
        _pandas_compression_type = 'gzip' if output_filepath_extension == '.gz' else None
        df = pd.DataFrame(smiles_processed, columns=["smiles"])
        df.to_csv(args.output_filepath, index=False, compression=_pandas_compression_type)
    np.save(args.output_properties_path, properties_processed)
    print(f"Data saved to {args.output_filepath}")
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    