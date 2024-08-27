import os
import argparse
import json
import numpy as np

from jointformer.utils.runtime import set_seed, create_output_dir, set_to_dev_mode, log_args, dump_configs
from jointformer.utils.ddp import init_ddp, end_ddp
from jointformer.utils.data import write_dict_to_file 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='results')
    parser.add_argument("--metric", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    results = {}
    fractions_train_dataset = os.listdir(args.out_dir)
    for fraction_train_dataset in fractions_train_dataset:
        results[fraction_train_dataset] = {}
        path = os.path.join(args.out_dir, fraction_train_dataset)
        dataset_seeds = os.listdir(path)
        for dataset_seed in dataset_seeds:
            path = os.path.join(args.out_dir, fraction_train_dataset, dataset_seed)
            with open(os.path.join(path, 'test_loss.json'), 'r') as file:
              data = json.load(file)
            results[fraction_train_dataset][dataset_seed] = data[args.metric]      
    return results

def aggregate_results(results):
    out = {}
    for key, value in results.items():
      out[key] = {}
      result = np.array(list(value.values()))
      out[key]['mean'] = round(np.mean(result), 3)
      out[key]['se'] = round(np.std(result, ddof=1) / np.sqrt(len(result)), 3) if len(result) > 1 else 0.000
    return out

if __name__ == "__main__":
    args = parse_args()
    write_dict_to_file(aggregate_results(main(args)), os.path.join(args.out_dir, 'aggregated_results.json'))
