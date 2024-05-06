""" Python script for distributed training of joint models. """

import os
import argparse

from torch.distributed.elastic.multiprocessing.errors import record
from jointformer.utils.loggers.wandb import WandbLogger

from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel

from jointformer.trainers.trainer import Trainer

from jointformer.utils.utils import set_seed

DEFAULT_SEED_ARRAY = [1337]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, nargs='*', default=DEFAULT_SEED_ARRAY)
    parser.add_argument("--path_to_task_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    # parser.add_argument("-logger", "--path_to_logger_config", type=str, required=True)
    args = parser.parse_args()
    return args


def create_output_dir(out_dir):
    # out_dir/dataset_name/target_label/num_samples/model_name/model_caption/task_p/seed_0
    # make lowercase
    os.makedirs(out_dir, exist_ok=True)


def main():

    args = parse_args()

    for seed in args.seed:
        set_seed(seed)



    # Create output directory
    # os.makedirs(args.out_dir, exist_ok=True)
    #
    # # Reproducibility
    # set_seed()

    #
    # # Load task, model, trainer and logger configurations
    # task_config = TaskConfig.from_json(args.path_to_task_config)
    # model_config = ModelConfig.from_json(args.path_to_model_config)
    # trainer_config = TrainerConfig.from_json(args.path_to_trainer_config)
    # logger_config = LoggerConfig.from_json(args.path_to_logger_config)
    #
    # # Initialize logger
    # logger = WandbLogger(logger_config)
    #
    # # Initialize tokenizer
    # tokenizer = AutoTokenizer(model_config)
    #
    # # Initialize dataset
    # dataset = AutoDataset(task_config, tokenizer)
    #
    # # Initialize model
    # model = AutoModel(model_config, dataset)
    #
    # # Initialize trainer
    # trainer = Trainer(model, dataset, trainer_config, logger)
    #
    # # Train model
    # trainer.train()


if __name__ == "__main__":
    main()
