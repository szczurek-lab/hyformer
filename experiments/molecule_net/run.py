import os
import sys
import argparse
import optuna
import logging

from socket import gethostname
from functools import partial

from jointformer.utils.optuna import get_hparam_search_space, load_json, save_json

from experiments.data_efficient_domain_adaptation.train import main as model_training_loop
from experiments.data_efficient_domain_adaptation.test import main as model_testing_loop

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", "0"))}: %(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--fraction_train_dataset", type=float, default=1.)
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--destroy_ckpt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--best_hparams_filename", type=str, default="best_hparams.json")
    args = parser.parse_args()
    return args


def load_best_params(args):
    console.info(f"Hparams loaded from {args.best_hparams_filename}.")
    return load_json(os.path.join(args.out_dir, args.best_hparams_filename))


def main(args):

    # Create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)

    # Load best hyperparameters
    console.info("Loading best hyperparameters from file...")
    hparams = load_best_params(args)

    # Train and Test model
    val_loss = model_training_loop(args, hparams)
    console.info(f"Best validation loss: {val_loss}")
    test_loss = model_testing_loop(args, hparams)
    console.info(f"Test loss: {test_loss}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
