import torch
from jointformer.utils.properties.auto import AutoTarget
import argparse
import multiprocessing as mp
import numpy as np
import os
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["qed", "plogp"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=1)
    return parser

def worker(oracle, chunk, id, ret_dict):
    ret_dict[id] = oracle(chunk)

def main(args):
    dname = os.path.dirname(args.output)
    os.makedirs(dname, exist_ok=True)

    oracle = AutoTarget.from_target_label(args.target)

    with open(args.data_path) as f:
        data = f.readlines()
    chunk_size = math.ceil(len(data) / args.n_workers)
    manager = mp.Manager()
    ret_dict = manager.dict()
    processes = []
    for i in range(args.n_workers):
        chunk = data[i*chunk_size: i*chunk_size + chunk_size]
        p = mp.Process(target=worker, args=(oracle, chunk, i, ret_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    ret = torch.cat([v for _, v in sorted(ret_dict.items())], dim=0)
    np.save(args.output, ret.numpy())
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)