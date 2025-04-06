import os
import sys
import torch
import logging

from socket import gethostname

from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.trainer import TrainerConfig
from hyformer.configs.logger import LoggerConfig

from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.models.auto import AutoModel
from hyformer.utils.loggers.auto import AutoLogger

from hyformer.trainers.trainer import Trainer

from hyformer.utils.experiments import set_seed, dump_configs

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", "0"))}: %(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args, hparams=None, disable_logging=False, max_iters=None):

    # Set seed
    set_seed(1337)

    # Disable logging for infer_max_num_epochs
    if max_iters is not None and max_iters == 'infer':
        disable_logging = True
    
    # Load configs
    dataset_config = DatasetConfig.from_config_path(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_path(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_path(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_path(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_path(args.path_to_logger_config) if args.path_to_logger_config else None

    # Adjust filepats for varying dataset seeds
    for key, value in dataset_config.__dict__.items():
        if value is not None and isinstance(value, str) and 'seed_0' in value:
            dataset_config[key] = value.replace('seed_0', f'seed_{args.seed}')
            print(f"Updated {key} to {dataset_config[key]}")

    # Init
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    trainer_config.correct_for_num_train_examples(num_train_examples=len(train_dataset))  # adjust trainer config to dataset size

    # Infer max_iters
    if max_iters is not None and isinstance(max_iters, int):
        console.info(f"Inferred max_iters equal to {max_iters}")
        max_iters = round(max_iters * (len(train_dataset + val_dataset) / len(train_dataset)))
        console.info(f"Setting max_iters to {max_iters}")
        train_dataset = train_dataset + val_dataset
        val_dataset = None
        trainer_config.max_iters = max_iters

    # If, debug
    if args.debug:
        console.info("Debugging...")
        trainer_config.max_iters = 2
        trainer_config.batch_size = 2
        trainer_config.eval_iters = 1
        trainer_config.eval_interval = 1
        trainer_config.log_interval = 1

    # Dump configs
    if not disable_logging:
        if args.path_to_model_ckpt is None:
            model_config.path_to_model_ckpt = args.path_to_model_ckpt
        dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    model = AutoModel.from_config(model_config, downstream_task=dataset_config.prediction_task_type, num_prediction_tasks=dataset_config.num_prediction_tasks, hidden_dim=256)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config)
    device = torch.device('cuda:0')
    trainer = Trainer(
        out_dir=None if disable_logging else args.out_dir, seed=1337, config=trainer_config, model=model,
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=val_dataset,
        tokenizer=tokenizer, logger=logger, device=device, test_metric=None)

    if args.path_to_model_ckpt is not None:
        if not os.path.exists(args.path_to_model_ckpt):
            raise ValueError(f"Model checkpoint {args.path_to_model_ckpt} does not exist.")
        trainer.resume_from_file(args.path_to_model_ckpt)
        console.info(f"Resuming from {args.path_to_model_ckpt}")
    else:
        console.info("Training from scrach.")

    trainer.train()

    if max_iters is not None and max_iters == 'infer':
        return trainer._best_iter

    return trainer._optuna_loss
