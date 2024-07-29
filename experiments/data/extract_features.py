from argparse import ArgumentParser
from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from typing import Callable
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.utils.datasets.auto import AutoDataset
from jointformer.models.auto import AutoModel
import logging
import sys
import os
import numpy as np
from tqdm import tqdm
from jointformer.models.base import SmilesEncoder
import multiprocessing as mp

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--path_to_model_ckpt", type=str, required=True)
    parser.add_argument("--split", choices=["train", "test", "val"], required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_task_config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser

def worker(model_factory, chunk, id, ret_dict):
    model = model_factory()
    ret_dict[id] = model.encode(chunk)

def extract_features_parallel(model_factory: Callable[[], SmilesEncoder], data: list[str], output_path: str, chunk_size: int):
    dname = os.path.dirname(output_path)
    os.makedirs(dname, exist_ok=True)

    manager = mp.Manager()
    ret_dict = manager.dict()
    processes = []

    for i in tqdm(range(0, len(data), chunk_size), "Encoding chunks"):
        chunk = data[i:i+chunk_size]

        p = mp.Process(target=worker, args=(model_factory, chunk, i, ret_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    ret = np.concatenate([v for _, v in sorted(ret_dict.items())], axis=0)
    np.save(output_path, ret)

def extract_features(model: SmilesEncoder, data: list[str], output_path: str, chunk_size: int):
    dname = os.path.dirname(output_path)
    fname, ext = os.path.splitext(os.path.basename(output_path))
    os.makedirs(dname, exist_ok=True)
    for i in tqdm(range(0, len(data), chunk_size), "Encoding chunks"):
        chunk = data[i:i+chunk_size]
        features = model.encode(chunk)
        np.save(os.path.join(dname, f"{fname}_chunk_{i}{ext}"), features)

    files = [os.path.join(dname, x) for x in os.listdir(dname) if x.startswith(fname)]
    chunk_files = [
        (int(os.path.splitext(f[f.find(fname) + len(fname) + len("_chunk_"):])[0]), f)
        for f in files
    ]
    chunk_files.sort()
    rets = []
    for _, f in chunk_files:
        rets.append(np.load(f))
    ret = np.concatenate(rets, axis=0)
    np.save(output_path, ret)

    for f in files:
        os.remove(f)


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    task_config = TaskConfig.from_config_file(args.path_to_task_config)
    task_config.target_label = None
    tokenizer = AutoTokenizer.from_config(task_config)
    dataset = AutoDataset.from_config(task_config, args.split)
    def model_factory():
        model = AutoModel.from_config(model_config)
        model.load_pretrained(args.path_to_model_ckpt)
        model = model.to_smiles_encoder(tokenizer, args.batch_size, args.device)
        return model
    
    #extract_features(model_factory(), dataset.data, args.output, args.chunk_size)
    extract_features_parallel(model_factory, dataset.data, args.output, args.chunk_size)



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)