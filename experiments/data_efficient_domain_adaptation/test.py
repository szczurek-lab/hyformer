import os
import sys
import logging
import argparse

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

from jointformer.trainers.trainer import Trainer

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
    args = parser.parse_args()
    return args


def main(args):

    path_to_model_ckpt = os.path.join(args.out_dir, 'ckpt.pt')
    assert os.path.exists(path_to_model_ckpt), f"Model checkpoint not found: {path_to_model_ckpt}"

    # Load configs
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_file(args.path_to_logger_config) if args.path_to_logger_config else None

    # Init
    test_dataset = AutoDataset.from_config(dataset_config, split='test', data_dir=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    set_seed(args.model_seed)
    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    
    # Test
    trainer = Trainer(
        out_dir=args.out_dir, seed=args.model_seed, config=trainer_config, model=model,
        test_dataset=test_dataset, tokenizer=tokenizer, logger=logger)
    trainer._init_data_loaders()
    trainer.resume_from_file(path_to_model_ckpt)

    objective_metric = trainer.test(metric=args.metric)
    print(f"Test {args.metric}: {objective_metric}")
    if args.destroy_ckpt:
        os.remove(path_to_model_ckpt)
    return objective_metric


if __name__ == "__main__":
    args = parse_args()
    test_metric = main(args)
    write_dict_to_file({f'{args.metric}': test_metric}, os.path.join(args.out_dir, 'test_loss.json'))
           