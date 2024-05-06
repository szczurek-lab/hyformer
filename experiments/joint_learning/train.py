import os, sys
import argparse

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed import init_process_group, destroy_process_group


from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig

from jointformer.utils.utils import set_seed


from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.trainers.trainer import Trainer


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
    # parser.add_argument("-logger", "--path_to_logger_config", type=str, required=True)
    parser.add_argument("--path_to_pretrained", type=str)
    args = parser.parse_args()
    return args


def set_to_dev_mode(**kwargs):
    print("Dev mode is on")
    task_config = kwargs.get("task_config", None)
    model_config = kwargs.get("model_config", None)
    trainer_config = kwargs.get("trainer_config", None)
    logger_config = kwargs.get("logger_config", None)

    if task_config and hasattr(task_config, "num_samples"):
        task_config.num_train_samples = 2
    if trainer_config and hasattr(trainer_config, "batch_size"):
        trainer_config.batch_size = 2
    if model_config:
        if hasattr(model_config, "num_layers"):
            model_config.num_layers = 1
        if hasattr(model_config, "num_heads"):
            model_config.num_heads = 1
        if hasattr(model_config, "embedding_dim"):
            model_config.embedding_dim = 16

    if logger_config and hasattr(logger_config, "enable_wandb"):
        logger_config.enable_wandb = False


def create_output_dir(out_dir):
    if not os.path.isdir(out_dir):
        is_ddp = int(os.environ.get('RANK', -1)) != -1
        is_master_process = int(os.environ.get('RANK', -1)) == 0
        if is_master_process & is_ddp:
            os.makedirs(out_dir, exist_ok=True)
            print(f"Output directory {out_dir} created...")


def init_ddp():
    init_process_group(backend=DDP_BACKEND)
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"])) // to be set by the trainer


@record
def main(args):

    print()

    # Determine if DDP is enabled
    ddp = int(os.environ.get('RANK', -1)) != -1
    print(f"DDP: {ddp}")

    # Initialize DDP
    if ddp:
        init_ddp()

    # Create output directory
    create_output_dir(args.out_dir)

    # Load Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
    if args.dev_mode:
        set_to_dev_mode(task_config=task_config, model_config=model_config)

    # Load data, tokenizer and model
    train_dataset = AutoDataset.from_config(task_config, split='train')
    val_dataset = AutoDataset.from_config(task_config, split='val')
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)

    # Load trainer // ckpt needs to load automatically, if available and be saved automatically
    trainer = Trainer(
        out_dir=args.out_dir, init_from=args.path_to_pretrained, seed=args.seed, config=trainer_config, model=model)

    if ddp:
        destroy_process_group()

    #
    # # Load task, model, trainer and logger configurations
    # task_config = TaskConfig.from_json(args.path_to_task_config)
    # model_config = ModelConfig.from_json(args.path_to_model_config)
    # trainer_config = TrainerConfig.from_json(args.path_to_trainer_config)
    # logger_config = LoggerConfig.from_json(args.path_to_logger_config)
    #
    # # Initialize logger
    # logger = WandbLogger(logger_config)
    #
    # # Initialize tokenizer
    # tokenizer = AutoTokenizer(model_config)
    #
    # # Initialize dataset
    # dataset = AutoDataset(task_config, tokenizer)
    #
    # # Initialize model
    # model = AutoModel(model_config, dataset)
    #
    # # Initialize trainer
    # trainer = Trainer(model, dataset, trainer_config, logger)
    #
    # # Train model
    # trainer.train()


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seed:
        tmp_args = args
        tmp_args.seed = seed
        tmp_args.out_dir = os.path.join(args.out_dir, f"seed_{seed}")
        set_seed(seed)
        main(tmp_args)
