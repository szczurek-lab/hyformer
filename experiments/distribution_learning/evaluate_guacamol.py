""" Run the training script for the distribution learning task. 

Handles both single-GPU and DDP training.

"""

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
# logging.captureWarnings(False)


@record
def main(args):    

    # Create output directory
    if args.out_dir is not None and not os.path.exists(args.out_dir) and int(os.environ.get('LOCAL_RANK', 0)) == 0:
        os.makedirs(args.out_dir, exist_ok=False)

    # Load configurations
    dataset_config = DatasetConfig.from_config_path(args.dataset_config_path)
    tokenizer_config = TokenizerConfig.from_config_path(args.tokenizer_config_path)
    model_config = ModelConfig.from_config_path(args.model_config_path)
    trainer_config = TrainerConfig.from_config_path(args.trainer_config_path)
    logger_config = LoggerConfig.from_config_path(args.logger_config_path) if args.logger_config_path else None
    
    # Set debug mode
    if args.debug:
        trainer_config.max_epochs = 2
        trainer_config.log_interval = 1
        trainer_config.batch_size = 2
    
    # Store configs within the output directory, for reproducibility
    if args.out_dir is not None:
        dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config) 

    # Initialize
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    
    # Set debug mode
    if args.debug:
        train_dataset.data, train_dataset.target = train_dataset.data[:100], train_dataset.target[:100] if train_dataset.target is not None else None
        val_dataset.data, val_dataset.target = val_dataset.data[:100], val_dataset.target[:100] if val_dataset.target is not None else None
    
    # Store configs within the logger object, for reproducibility
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    # Determine the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Determine whether to use DDP
    if trainer_config.enable_ddp and int(os.environ.get('RANK', -1)) != -1 and torch.cuda.device_count() > 1:
        print("Running in distributed setting...", flush=True)
        init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}')
        seed = args.experiment_seed + int(os.environ['LOCAL_RANK'])
        print(f"Rank: {int(os.environ['RANK'])}, Local Rank: {int(os.environ['LOCAL_RANK'])}, Device: {device}", flush=True)

    # Initialize trainer
    trainer = Trainer(
        config=trainer_config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        out_dir=args.out_dir,
        logger=logger,
        worker_seed=args.experiment_seed
        )

    if args.path_to_model_ckpt:
        trainer.resume_from_checkpoint(args.path_to_model_ckpt, resume_training=args.resume_training)
        console.info(f"Resuming pre-trained model from {args.path_to_model_ckpt}")
    else:
        console.info("Training from scratch")

    # Ensure all processes are ready before training
    if trainer._is_distributed_run:
        dist.barrier() 

    # Run training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_specific_validation=args.task_specific_validation,
        patience=args.patience,
        )

    # Clean up
    if trainer_config.enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--experiment-seed", type=int, default=0, help="Seed for the experiment")
    parser.add_argument("--dataset-config-path", type=str, required=True, help="Path to the dataset config file")
    parser.add_argument("--tokenizer-config-path", type=str, required=True, help="Path to the tokenizer config file")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--trainer-config-path", type=str, required=True, help="Path to the trainer config file")
    parser.add_argument("--logger-config-path", type=str, nargs='?', help="Path to the logger config file")
    parser.add_argument("--model-ckpt-path", type=str, nargs='?', help="Path to the model checkpoint file")
    parser.add_argument("--resume-training", default=False, action=argparse.BooleanOptionalAction, help="Resume training from the checkpoint file")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction, help="Run in debug mode")
    parser.add_argument("--task-specific-validation", type=str, nargs='?', help="Task with respect to which validation is performed")
    parser.add_argument("--patience", type=int, nargs='?', help="Number of epochs to wait before early stopping")
    args = parser.parse_args()
    log_args(args)
    return args

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    args = parse_args()
    set_seed(args.experiment_seed)
    main(args)
    