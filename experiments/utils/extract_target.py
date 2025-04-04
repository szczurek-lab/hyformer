import argparse
import multiprocessing as mp
import numpy as np
import os
import math

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
    else:
        with open(args.data_filepath) as f:
            data = f.readlines()

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
    