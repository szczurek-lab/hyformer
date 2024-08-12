from argparse import ArgumentParser
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.dataset import DatasetConfig

from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.utils.datasets.auto import AutoDataset
from jointformer.models.auto import AutoModel
import logging
import sys
import os
import numpy as np
from tqdm import tqdm
from jointformer.models.base import SmilesEncoder

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--path_to_model_ckpt", type=str, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


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
    model = AutoModel.from_config(model_config)

    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    dataset_config.target_label = None
    dataset = AutoDataset.from_config(dataset_config, args.split)
    
    model.load_pretrained(args.path_to_model_ckpt)
    model = model.to_smiles_encoder(tokenizer, args.batch_size, args.device)
    
    extract_features(model, dataset.data, args.output, args.chunk_size)



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)