import os, logging, argparse, sys

import torch

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

from hyformer.utils.experiments import set_seed, log_args, dump_configs

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)

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
    parser.add_argument("--freeze_weights", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--patience", type=int, default=None, help="Patience for the model")
    parser.add_argument("--eval_metric", type=str, default=None, help="Evaluation metric for the model")
    args = parser.parse_args()
    log_args(args)
    return args


def main(args):    

    # Create output directory
    if args.out_dir is not None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)

    # Load configurations
    dataset_config = DatasetConfig.from_config_path(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_path(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_path(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_path(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_path(args.path_to_logger_config) if args.path_to_logger_config else None
    if args.out_dir is not None:
        dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config) # Store configs, within the out_dir

    # Initialize
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)    
    model = AutoModel.from_config(model_config, downstream_task=dataset_config.prediction_task_type, num_prediction_tasks=dataset_config.num_prediction_tasks)
    
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config) # Store configs, within the logger object

    # Freeze weights
    if args.freeze_weights:
        for name, param in model.named_parameters():
            if not name.startswith('prediction_head'):
                param.requires_grad = False
        print("Freezing weights...", flush=True)

    
    # Run training
    trainer = Trainer(
        out_dir=args.out_dir,
        seed=args.seed,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        logger=logger,
        test_metric=dataset_config.test_metric,
        eval_metric=args.eval_metric,
        patience=args.patience
        )

    if args.path_to_model_ckpt:
        trainer.resume_from_file(args.path_to_model_ckpt, resume_training=False)
        console.info(f"Loading pre-trained model from {args.path_to_model_ckpt}")
    else:
        console.info("Training from scratch")

    trainer.train()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    args = parse_args()
    set_seed(args.seed)
    main(args)
