import os, logging, argparse, sys

import torch
import torch.distributed as dist

from socket import gethostname
from torch.distributed.elastic.multiprocessing.errors import record

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
from torch.distributed import init_process_group, destroy_process_group

import torch
from scipy.stats import truncnorm


console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)


class MLMProbability:
    def __init__(self, mean=0.55, sigma=0.25, low=0.5, high=1.0):
        self.mean = mean
        self.sigma = sigma
        self.low = low
        self.high = high
        self.a = (low - mean) / sigma  # lower bound in standard normal space
        self.b = (high - mean) / sigma  # upper bound in standard normal space

    def __call__(self):
        # Draw from a truncated normal
        return torch.tensor(truncnorm.rvs(self.a, self.b, loc=self.mean, scale=self.sigma), dtype=torch.float).item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='./results')
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--resume_training", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    log_args(args)
    return args


@record
def main(args):    

    # Create output directory
    if args.out_dir is not None and not os.path.exists(args.out_dir) and int(os.environ.get('LOCAL_RANK', 0)) == 0:
        os.makedirs(args.out_dir, exist_ok=False)

    # Load configurations
    dataset_config = DatasetConfig.from_config_path(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_path(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_path(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_path(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_path(args.path_to_logger_config) if args.path_to_logger_config else None
    if args.out_dir is not None:
        dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config) # Store configs, within the out_dir
    
    if args.dry_run:
        trainer_config.max_epochs = 2
        trainer_config.eval_iters = 2
        trainer_config.log_interval = 1

    # Initialize
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    if tokenizer.mlm_probability == 'MAGE': 
        tokenizer.mlm_probability = MLMProbability()
    
    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config) # Store configs, within the logger object

    # Determine whether to use DDP
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    seed = args.seed
    if trainer_config.enable_ddp and int(os.environ.get('RANK', -1)) != -1 and torch.cuda.device_count() > 1:
        print("Running in distributed setting...", flush=True)
        init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}')
        seed = seed + int(os.environ['LOCAL_RANK'])
        print(f"Rank: {int(os.environ['RANK'])}, Local Rank: {int(os.environ['LOCAL_RANK'])}, Device: {device}", flush=True)

    # Run training
    trainer = Trainer(
        out_dir=args.out_dir,
        seed=seed,
        device=device,
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        logger=logger)

    try:
        trainer.resume_snapshot()
        console.info("Resumed Snapshot")
    except FileNotFoundError:
        if args.path_to_model_ckpt:
            trainer.resume_from_file(args.path_to_model_ckpt, resume_training=args.resume_training)
            console.info(f"Resuming pre-trained model from {args.path_to_model_ckpt}")
        else:
            console.info("Training from scratch")

    if trainer._is_distributed_run:
        dist.barrier() # Ensure all processes are ready before training

    trainer.train()

    if trainer_config.enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    args = parse_args()
    main(args)
