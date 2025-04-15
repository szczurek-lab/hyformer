import argparse
import multiprocessing as mp
import numpy as np
import os
import math
import lmdb
import pickle
from tqdm import tqdm

from hyformer.utils.targets.auto import AutoTarget


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name", choices=["qed", "plogp", "guacamol_mpo", "physchem"], required=True)
    parser.add_argument("--data_filepath", type=str, required=True)
    parser.add_argument("--target_filepath", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    return parser


def worker(oracle, chunk, id, ret_dict):
    ret_dict[id] = oracle(chunk)


def main(args):
    dname = os.path.dirname(args.target_filepath)
    os.makedirs(dname, exist_ok=True)

    oracle = AutoTarget.from_target_name(args.target_name)

    if os.path.splitext(args.data_filepath)[1] == '.npy':
        data = np.load(args.data_filepath, allow_pickle=True)
    elif os.path.splitext(args.data_filepath)[1] == '.lmdb':
        env = lmdb.open(args.data_filepath, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
        txn = env.begin()
        keys = list(txn.cursor().iternext(values=False))
        data = []
        for idx in tqdm(keys, desc="Loading .lmdb data"):
            datapoint_pickled = txn.get(idx)
            datapoint = pickle.loads(datapoint_pickled)
            data.append(datapoint['smi'])
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
    