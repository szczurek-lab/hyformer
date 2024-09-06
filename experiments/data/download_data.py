import os
import logging
import argparse

from tqdm.contrib.logging import logging_redirect_tqdm

from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.runtime import set_seed

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    filename="data.log",
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logging.captureWarnings(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--path_to_task_config", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):

    # Seed
    set_seed(args.seed)

    # Config
    dataset_config = DatasetConfig.from_config_file(args.path_to_task_config)
    
    # Data
    train_dataset = AutoDataset.from_config(dataset_config, split='train', out_dir=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', out_dir=args.data_dir)
    test_dataset = AutoDataset.from_config(dataset_config, split='test', out_dir=args.data_dir)
    

if __name__ == "__main__":
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, f"seed_{args.seed}")
    try:
        with logging_redirect_tqdm():
            main(args)
    except Exception as e:
        logging.critical(e, exc_info=True)
