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

from hyformer.trainers.trainer_fixed import Trainer

from hyformer.utils.runtime import set_seed, dump_configs

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
    # if max_iters is not None and max_iters == 'infer':
    #     disable_logging = True
    
    # Load configs
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_file(args.path_to_logger_config) if args.path_to_logger_config else None

    # Adjust filepats for varying dataset seeds
    for key, value in dataset_config.__dict__.items():
        if value is not None and isinstance(value, str) and 'seed_0' in value:
            dataset_config[key] = value.replace('seed_0', f'seed_{args.seed}')
            print(f"Updated {key} to {dataset_config[key]}")

    if hasattr(args, 'decay_lr') and args.decay_lr is not None:
        trainer_config.decay_lr = args.decay_lr
        print("Decay learning rate updated to", trainer_config.decay_lr)
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        trainer_config.batch_size = args.batch_size
        print("Batch size updated to", trainer_config.batch_size)
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        trainer_config.learning_rate = args.learning_rate
        trainer_config.min_lr = 0.1 * args.learning_rate
        print("Learning rate updated to", trainer_config.learning_rate)
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        trainer_config.weight_decay = args.weight_decay
        print("Weight decay updated to", trainer_config.weight_decay)
    if hasattr(args, 'pooler_dropout') and args.pooler_dropout is not None:
        model_config.pooler_dropout = args.pooler_dropout
        print("Pooler dropout updated to", model_config.pooler_dropout)
    if hasattr(args, 'max_epochs') and args.max_epochs is not None:
        trainer_config.max_epochs = args.max_epochs
        print("Max epochs updated to", trainer_config.max_epochs)

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

    model = AutoModel.from_config(model_config, downstream_task=dataset_config.task_type, num_tasks=dataset_config.num_tasks, hidden_dim=256)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config)
    device = torch.device('cuda:0')
    trainer = Trainer(
        out_dir=None if disable_logging else args.out_dir, seed=1337, config=trainer_config, model=model,
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=val_dataset,
        tokenizer=tokenizer, logger=logger, device=device, test_metric=None, patience=args.patience)

    if args.path_to_model_ckpt is not None:
        if not os.path.exists(args.path_to_model_ckpt):
            raise ValueError(f"Model checkpoint {args.path_to_model_ckpt} does not exist.")
        trainer.resume_from_file(args.path_to_model_ckpt)
        console.info(f"Resuming from {args.path_to_model_ckpt}")
    else:
        console.info("Training from scrach.")

    trainer.train()

    if max_iters is not None and max_iters in ['infer', 'infer_generation']:
        return trainer._best_iter

    return trainer._optuna_loss
