""" Run the training script for the distribution learning task. 

Handles both single-GPU and DDP training.
"""

import os, logging, argparse, sys

import torch
import torch.distributed as dist

from torch.distributed import init_process_group, destroy_process_group
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

from hyformer.utils.experiments import log_args, dump_configs
from hyformer.utils.reproducibility import set_seed

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
        if args.debug:
            os.path.join(args.out_dir, "debug")
        os.makedirs(args.out_dir, exist_ok=False)

    # Load configurations
    dataset_config = DatasetConfig.from_config_path(args.dataset_config_path)
    tokenizer_config = TokenizerConfig.from_config_path(args.tokenizer_config_path)
    model_config = ModelConfig.from_config_path(args.model_config_path)
    trainer_config = TrainerConfig.from_config_path(args.trainer_config_path)
    logger_config = LoggerConfig.from_config_path(args.logger_config_path) if args.logger_config_path else None
    
    # Set debug mode
    if args.debug:
        model_config.num_transformer_layers = 2
        trainer_config.max_epochs = 2
        trainer_config.log_interval = 1
        trainer_config.batch_size = 2
        trainer_config.warmup_iters = 10
    
    # Set learning rate
    if args.learning_rate is not None:
        trainer_config.learning_rate = args.learning_rate
        console.info(f"Learning rate set to: {args.learning_rate}")
    
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
        train_dataset.data, train_dataset.target = train_dataset.data[:1500], train_dataset.target[:1500] if train_dataset.target is not None else None
        val_dataset.data, val_dataset.target = val_dataset.data[:1500], val_dataset.target[:1500] if val_dataset.target is not None else None
    
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

    if args.model_ckpt_path:
        trainer.resume_from_checkpoint(args.model_ckpt_path, resume_training=args.resume_training)
        console.info(f"Resuming pre-trained model from {args.model_ckpt_path}")
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
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--experiment_seed", type=int, default=0, help="Seed for the experiment")
    parser.add_argument("--dataset_config_path", type=str, required=True, help="Path to the dataset config file")
    parser.add_argument("--tokenizer_config_path", type=str, required=True, help="Path to the tokenizer config file")
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--trainer_config_path", type=str, required=True, help="Path to the trainer config file")
    parser.add_argument("--logger_config_path", type=str, nargs='?', help="Path to the logger config file")
    parser.add_argument("--model_ckpt_path", type=str, nargs='?', help="Path to the model checkpoint file")
    parser.add_argument("--resume_training", default=False, action=argparse.BooleanOptionalAction, help="Resume training from the checkpoint file")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction, help="Run in debug mode")
    parser.add_argument("--task_specific_validation", type=str, nargs='?', help="Task with respect to which validation is performed")
    parser.add_argument("--patience", type=int, nargs='?', help="Number of epochs to wait before early stopping")
    parser.add_argument("--use_deterministic_algorithms", default=False, action=argparse.BooleanOptionalAction, help="Use deterministic algorithms")
    parser.add_argument("--learning_rate", type=float, nargs='?', help="Learning rate")
    args = parser.parse_args()
    log_args(args)
    return args

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    args = parse_args()
    set_seed(args.experiment_seed, use_deterministic_algorithms=args.use_deterministic_algorithms)
    main(args)
    