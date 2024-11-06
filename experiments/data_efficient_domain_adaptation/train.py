import os
import sys
import torch
import logging
import argparse

from socket import gethostname
import torch.nn as nn

from jointformer.configs.dataset import DatasetConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.auto import AutoLogger

from jointformer.trainers.trainer import Trainer

from jointformer.utils.runtime import set_seed, create_output_dir, set_to_dev_mode, log_args, dump_configs
from jointformer.utils.ddp import init_ddp, end_ddp
from jointformer.utils.data import write_dict_to_file


console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", "0"))}: %(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_seed", type=int, required=True)
    parser.add_argument("--model_seed", type=int, required=True)
    args = parser.parse_args()
    return args


def main(args, hparams=None, disable_logging=False):

    if not os.path.exists(args.out_dir) and not disable_logging:
        os.makedirs(args.out_dir, exist_ok=False)

    assert torch.cuda.is_available(), "CUDA is not available"
    set_seed(1336 + args.seed)

    # Configs
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_file(args.path_to_logger_config) if args.path_to_logger_config else None

    # Load trainer hparams
    if hparams is not None:
        for key, value in hparams.items():
            if key in model_config.__dict__.keys():
                model_config[key] = value
            if key in trainer_config.__dict__.keys():
                trainer_config[key] = value
            if key == 'generation_task':
                trainer_config['tasks'] = {"prediction": 20, "generation": value}
                trainer_config._normalize_task_probabilities()
            if key == 'model_seed':
                args.model_seed = value

    # Init
    train_dataset = AutoDataset.from_config(dataset_config, split='train', data_dir=args.data_dir)
    # num_subsamples =  int(len(train_dataset) * args.fraction_train_dataset)
    # train_dataset._subset(num_samples=num_subsamples, seed=args.seed)
    # console.info(f"Selected {len(train_dataset)} training examples")
    val_dataset = AutoDataset.from_config(dataset_config, split='val', data_dir=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    # Adjust trainer config
    if disable_logging:
        trainer_config.max_epochs = 10
    else:
        trainer_config.max_epochs = 20
    trainer_config.correct_for_num_train_examples(num_train_examples=len(train_dataset))  # adjust trainer config to dataset size

    # Test
    if args.test:
        console.info("Running in test mode")
        trainer_config.max_iters = 2
        trainer_config.batch_size = 2
        trainer_config.eval_iters = 1
        trainer_config.eval_interval = 1
        trainer_config.log_interval = 1

    # Dump configs
    if not disable_logging:
        dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    model = AutoModel.from_config(model_config, downstream_task=dataset_config.task_type, num_tasks=dataset_config.num_tasks, hidden_dim=256)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    trainer = Trainer(
        out_dir=None if disable_logging else args.out_dir, seed=args.model_seed, config=trainer_config, model=model,
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=val_dataset,
        tokenizer=tokenizer, logger=logger)

    if args.path_to_model_ckpt is not None and os.path.exists(args.path_to_model_ckpt):
        trainer.resume_from_file(args.path_to_model_ckpt)
        console.info(f"Resuming from {args.path_to_model_ckpt}")
    else:
        console.info("Training from scrach.")

    trainer.train()
    return trainer._optuna_loss


if __name__ == "__main__":
    args = parse_args()
    main(args)
           