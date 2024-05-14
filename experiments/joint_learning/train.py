import os, sys
import argparse

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed import init_process_group, destroy_process_group


from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.auto import AutoLogger

from jointformer.trainers.trainer import Trainer

from jointformer.utils.utils import set_seed

DEFAULT_SEED_ARRAY = [1337]
DDP_BACKEND = "nccl"
TORCHELASTIC_ERROR_FILE = "torchelastic_error_file.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, nargs='*', default=DEFAULT_SEED_ARRAY)
    parser.add_argument("--dev_mode", nargs='*', default=False)
    parser.add_argument("--path_to_task_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, required=True)
    parser.add_argument("--path_to_pretrained", type=str)
    args = parser.parse_args()
    return args


def set_to_dev_mode(**kwargs):
    print("Dev mode is on")
    task_config = kwargs.get("task_config", None)
    model_config = kwargs.get("model_config", None)
    trainer_config = kwargs.get("trainer_config", None)
    logger_config = kwargs.get("logger_config", None)

    if task_config:
        if hasattr(task_config, "num_samples"):
            task_config.num_samples = 4
    if model_config:
        if hasattr(model_config, "num_layers"):
            model_config.num_layers = 1
        if hasattr(model_config, "num_heads"):
            model_config.num_heads = 1
        if hasattr(model_config, "embedding_dim"):
            model_config.embedding_dim = 16
    if trainer_config and hasattr(trainer_config, "batch_size"):
        trainer_config.batch_size = 2
        trainer_config.max_iters = 1000
        trainer_config.eval_every = 500
        trainer_config.eval_iters = 2
    if logger_config and hasattr(logger_config, "enable_wandb"):
        logger_config.display_name = 'test'


def create_output_dir(out_dir):
    if not os.path.isdir(out_dir):
        is_ddp = int(os.environ.get('RANK', -1)) != -1
        is_master_process = int(os.environ.get('RANK', -1)) == 0
        if is_master_process & is_ddp:
            os.makedirs(out_dir, exist_ok=True)
            print(f"Output directory {out_dir} created...")


def init_ddp():
    init_process_group(backend=DDP_BACKEND)


@record
def main(args):

    # Load Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
    if args.dev_mode:
        set_to_dev_mode(task_config=task_config, model_config=model_config)
    logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

    # Initialize DDP
    is_ddp_run = int(os.environ.get('RANK', -1)) != -1 and trainer_config.enable_ddp
    print(f"DDP: {is_ddp_run}")
    if is_ddp_run:
        init_ddp()

    # Create output directory
    create_output_dir(args.out_dir)

    # Load data, tokenizer and model
    train_dataset = AutoDataset.from_config(task_config, split='train')
    val_dataset = AutoDataset.from_config(task_config, split='val')
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config, display_name='test')

    logger.store_config(task_config, model_config, trainer_config, logger_config)
    logger.save_config(args.out_dir)

    trainer = Trainer(
        out_dir=args.out_dir,
        init_from=args.path_to_pretrained,
        seed=args.seed,
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        logger=logger
    )
    trainer.train()

    if is_ddp_run:
        destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seed:
        tmp_args = args
        tmp_args.seed = seed
        tmp_args.out_dir = os.path.join(args.out_dir, f"seed_{seed}")
        set_seed(seed)
        main(tmp_args)
