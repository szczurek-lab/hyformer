import argparse
import os

import datamol as dm
import numpy as np
import pandas as pd

from typing import List, Union, Optional, Tuple
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

from jointformer.utils.properties.auto import AutoTarget


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--add_descriptors", type=str, nargs='?', const=True, default=False)
    parser.add_argument("--test_size", type=int, nargs='?', const=True, default=False)
    return parser


def _preprocess(smiles: str, max_length: int, oracle: Optional[object] = None) -> Union[None, str, Tuple[str, np.ndarray]]:

    _DISABLE_RDKIT_LOGS = True
    with dm.without_rdkit_log(enable=_DISABLE_RDKIT_LOGS):
        
        try:
            # mol = dm.to_mol(smiles, ordered=True)
            # mol = dm.fix_mol(mol)
            # mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
            # mol = dm.standardize_mol(
            #     mol,
            #     disconnect_metals=False,
            #     normalize=True,
            #     reionize=True,
            #     uncharge=False,
            #     stereo=True,
            # )
            # standardized_smile = dm.standardize_smiles(dm.to_smiles(mol))
            
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

            if oracle:
                try:
                    descriptor = oracle(standardized_smile)
                    return standardized_smile, descriptor
                except:
                    return standardized_smile, None
        except:
            return None
        
    return standardized_smile


def _worker(args):
    # Wrapper to unpack arguments for multiprocessing
    return _preprocess(*args)


def preprocess(smiles: List[str], max_length: int, n_jobs: Optional[int] = -1, add_descriptors: Optional[str] = None) -> List[str]:
    """
    Parallelized preprocessing of a list of SMILES strings.
    """

    if add_descriptors:
        oracle = AutoTarget.from_target_label(add_descriptors, dtype='np')
        oracle.verbose = False
    else:
        oracle = None
    
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    with Pool(n_jobs) as pool:
        # Prepare arguments for _preprocess_single
        args = [(smile, max_length, oracle) for smile in smiles]
        results = list(
            tqdm(
                pool.imap(_worker, args),
                total=len(smiles),
                desc="Preprocessing SMILES strings"
            )
        )

    if add_descriptors:
        results = [res for res in results if res is not None]
        results = [res for res in results if res[0] is not None and np.all(res[1] == res[1])]
        results, descriptors = zip(*results)
        print("Descriptors extracted for", len(results), "SMILES strings.")
        return results, np.vstack(descriptors)
    # Filter out None results (if a SMILES string was not valid or too long)
    return [res for res in results if res is not None]
    
    
def main(args):

    dname = os.path.dirname(args.output)
    os.makedirs(dname, exist_ok=True)

    if os.path.splitext(args.data_path)[1] == '.gz':
        data = pd.read_csv(args.data_path, compression='gzip', header=None)[0].values
    elif os.path.splitext(args.data_path)[1] == '.npy':
        data = np.load(args.data_path, allow_pickle=True).tolist()
    else:
        raise ValueError("Data format not supported.")
    
    data_processed = preprocess(data, max_length=args.max_length, n_jobs=args.n_workers, add_descriptors=args.add_descriptors)

    if args.add_descriptors:
        if args.test_size:
            from sklearn.model_selection import train_test_split
            SEED = 0
            test_data_size = float(int(args.test_size / len(data_processed[0]))) if args.test_size < len(data_processed[0]) else 0.1
            train_data, test_data, train_descriptors, test_descriptors = train_test_split(data_processed[0], data_processed[1], test_size=test_data_size, random_state=SEED)
            np.save(args.output.replace('.npy', '_train.npy'), train_data)
            np.save(args.output.replace('.npy', '_test.npy'), test_data)
            np.save(args.output.replace('.npy', f'_{args.add_descriptors}_train.npy'), train_descriptors)
            np.save(args.output.replace('.npy', f'_{args.add_descriptors}_test.npy'), test_descriptors)
        else:                
            np.save(args.output, data_processed[0])
            np.save(args.output.replace('.npy', f'_{args.add_descriptors}.npy'), data_processed[1])
    
    else:
        np.save(args.output, np.array(data_processed))
    print(f"Data saved to {args.output}")
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    