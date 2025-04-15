import argparse
import multiprocessing as mp
import numpy as np
import os
import math
import lmdb
import pickle
from tqdm import tqdm
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count
from functools import partial

from hyformer.utils.targets.auto import AutoTarget


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name", choices=["qed", "plogp", "guacamol_mpo", "physchem"], required=True)
    parser.add_argument("--data_filepath", type=str, required=True)
    parser.add_argument("--target_filepath", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--num_workers_lmdb', type=int, default=None, help='Number of worker processes for LMDB')
    return parser


def worker(oracle, chunk, id, ret_dict):
    ret_dict[id] = oracle(chunk)


def process_batch(batch: List[bytes], batch_size: int) -> List[Dict[str, Any]]:
    """Process a batch of pickled data in parallel."""
    results = []
    for datapoint_pickled in batch:
        try:
            datapoint = pickle.loads(datapoint_pickled)
            results.append(datapoint)
        except:
            continue
    return results


def load_lmdb_data(filepath: str, batch_size: int = 1000, num_workers: int = None) -> List[Dict[str, Any]]:
    """Load data from LMDB database efficiently using parallel processing."""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    env = lmdb.open(
        filepath,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256
    )
    
    txn = env.begin()
    cursor = txn.cursor()
    
    # Get all keys first
    keys = list(cursor.iternext(values=False))
    total_keys = len(keys)
    
    # Process in batches
    data = []
    with Pool(num_workers) as pool:
        for i in tqdm(range(0, total_keys, batch_size), desc="Loading batches"):
            batch_keys = keys[i:i + batch_size]
            batch_data = [txn.get(key) for key in batch_keys]
            
            # Process batch in parallel
            batch_results = pool.map(partial(process_batch, batch_size=batch_size), [batch_data])
            
            # Flatten results
            for result in batch_results:
                data.extend(result)
    
    return data


def main(args):
    dname = os.path.dirname(args.target_filepath)
    os.makedirs(dname, exist_ok=True)

    oracle = AutoTarget.from_target_name(args.target_name)

    if os.path.splitext(args.data_filepath)[1] == '.npy':
        data = np.load(args.data_filepath, allow_pickle=True)
    elif os.path.splitext(args.data_filepath)[1] == '.lmdb':
        data = load_lmdb_data(args.data_filepath, args.batch_size, args.num_workers_lmdb)
    else:
        with open(args.data_filepath) as f:
            data = f.readlines()
        data = [line.strip() for line in data]

    assert len(data) > 0, "No data found"
    print(f"Loaded {len(data)} datapoints from file {args.data_filepath}")
    try:
        np.save(os.path.splitext(args.data_filepath)[0] + '.npy', data)
    except Exception as e:
        print(f"Error saving data: {e}")
        print(f"Data shape: {np.array(data).shape}")
        print(f"Data type: {type(data[0])}")
        print(f"Data length: {len(data)}")
        raise e
    
    chunk_size = math.ceil(len(data) / args.num_workers)
    manager = mp.Manager()
    ret_dict = manager.dict()
    processes = []
    for i in range(args.num_workers):
        chunk = data[i*chunk_size: i*chunk_size + chunk_size]
        p = mp.Process(target=worker, args=(oracle, chunk, i, ret_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    ret = np.concatenate([v for _, v in sorted(ret_dict.items())], axis=0)
    np.save(args.target_filepath, ret)
    assert len(ret.shape) == 2, f"Expected 2D array, got {len(ret.shape)}D array"

    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    