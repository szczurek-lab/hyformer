import os
import sys
import logging
import argparse
import torch 
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

from hyformer.utils.experiments import set_seed, create_output_dir, set_to_dev_mode, log_args, dump_configs
from hyformer.utils.ddp import init_ddp, end_ddp
from hyformer.utils.file_io import write_dict_to_file


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
    parser.add_argument("--out_dir", type=str, default='results')
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--destroy_ckpt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--adjust_dataset_seed", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


def main(args, hparams=None):

    # Set seed
    set_seed(1337)
    path_to_model_ckpt = os.path.join(args.out_dir, 'ckpt.pt')

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

    # Load trainer hparams
    if hparams is not None:
        print("Updating hparams")
        for key, value in hparams.items():
            if key in model_config.__dict__.keys():
                model_config[key] = value
            if key in trainer_config.__dict__.keys():
                trainer_config[key] = value

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
    test_dataset = AutoDataset.from_config(dataset_config, split='test', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    model = AutoModel.from_config(model_config, downstream_task=dataset_config.prediction_task_type, num_prediction_tasks=dataset_config.num_prediction_tasks, hidden_dim=256)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    
    # Test
    device = torch.device('cuda:0')
    trainer = Trainer(
        out_dir=args.out_dir, config=trainer_config, model=model,
        test_dataset=test_dataset, tokenizer=tokenizer, logger=logger, seed=1337, device=device, test_metric=dataset_config.test_metric)
    trainer._init_data_loaders()
    print(f"Loading model from {path_to_model_ckpt}")
    trainer.model.load_state_dict(torch.load(path_to_model_ckpt, map_location=device)['model'], strict=True)

    test_metric = dataset_config.test_metric
    objective_metric = trainer.test(metric=test_metric)
    print(f"Test {test_metric}: {objective_metric}")
    write_dict_to_file({f'{test_metric}': str(objective_metric)}, os.path.join(args.out_dir, 'test_loss.json'))
    
    return objective_metric


if __name__ == "__main__":
    args = parse_args()
    main(args)
           