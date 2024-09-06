import os
import argparse
import json
import numpy as np
import logging
from socket import gethostname

from jointformer.utils.runtime import set_seed, create_output_dir, set_to_dev_mode, log_args, dump_configs
from jointformer.utils.ddp import init_ddp, end_ddp
from jointformer.utils.data import write_dict_to_file 


console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    filename=f"{os.environ.get('SLURM_JOB_NAME', 'run')}.log",
    filemode='a',
    format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", 0))}: %(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)

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
        if os.path.isdir(os.path.join(args.out_dir, fraction_train_dataset)):
            results[fraction_train_dataset] = {}
            path = os.path.join(args.out_dir, fraction_train_dataset)
            dataset_seeds = os.listdir(path)
            for dataset_seed in dataset_seeds:
                if os.path.isdir(os.path.join(args.out_dir, fraction_train_dataset, dataset_seed)):
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
    results = aggregate_results(main(args))
    write_dict_to_file(results, os.path.join(args.out_dir, 'aggregated_results.json'))
    print(results)
    