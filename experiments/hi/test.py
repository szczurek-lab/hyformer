import os
import sys
import logging
import argparse
import torch 
from socket import gethostname

from jointformer.configs.dataset import DatasetConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.auto import AutoLogger

from jointformer.trainers.trainer_fixed import Trainer

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

    # Init
    test_dataset = AutoDataset.from_config(dataset_config, split='test', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    model = AutoModel.from_config(model_config, downstream_task=dataset_config.task_type, num_tasks=dataset_config.num_tasks, hidden_dim=256)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    
    # Test
    device = torch.device('cuda:0')
    trainer = Trainer(
        out_dir=args.out_dir, config=trainer_config, model=model,
        test_dataset=test_dataset, tokenizer=tokenizer, logger=logger, seed=1337, device=device, test_metric=dataset_config.task_metric)
    trainer._init_data_loaders()
    print(f"Loading model from {path_to_model_ckpt}")
    trainer.model.load_state_dict(torch.load(path_to_model_ckpt, map_location=device)['model'], strict=True)

    test_metric = dataset_config.task_metric
    objective_metric = trainer.test(metric=test_metric)
    print(f"Test {test_metric}: {objective_metric}")
    write_dict_to_file({f'{test_metric}': str(objective_metric)}, os.path.join(args.out_dir, 'test_loss.json'))
    
    return objective_metric


if __name__ == "__main__":
    args = parse_args()
    main(args)
           