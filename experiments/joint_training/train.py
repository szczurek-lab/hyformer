import os, logging, argparse, time

from torch.distributed.elastic.multiprocessing.errors import record

from jointformer.configs.task import TaskConfig
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

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    filename="job.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logging.captureWarnings(True)


DEFAULT_SEED_ARRAY = [1337]
DDP_BACKEND = "nccl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='./')
    parser.add_argument("--path_to_task_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--pretrained_filename", type=str, nargs='?')
    parser.add_argument("--logger_display_name", nargs='?', type=str)
    parser.add_argument("--seed", type=int, nargs='*', default=DEFAULT_SEED_ARRAY)
    parser.add_argument("--dev_mode", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    log_args(args)
    return args


@record
def main(args):

    # Load Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config) if args.path_to_logger_config else None

    # Dev mode
    if args.dev_mode:
        set_to_dev_mode(
            task_config=task_config, model_config=model_config,
            trainer_config=trainer_config, logger_config=logger_config
        )

    # Initialize DDP
    init_ddp(trainer_config.enable_ddp, DDP_BACKEND)

    # Create output directory
    create_output_dir(args.out_dir)

    # Load data, tokenizer and model
    train_dataset = AutoDataset.from_config(task_config, split='train', out_dir=args.out_dir)
    val_dataset = AutoDataset.from_config(task_config, split='val', out_dir=args.out_dir)
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None

    # Store configs
    dump_configs(args.out_dir, task_config, model_config, trainer_config, logger_config)
    if logger is not None:
        logger.store_configs(task_config, model_config, trainer_config, logger_config)
        if args.logger_display_name is not None:
            logger.set_display_name(args.logger_display_name)

    trainer = Trainer(
        out_dir=os.path.join(args.out_dir, 'results'),
        seed=args.seed,
        config=trainer_config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        logger=logger
    )

    try:
        trainer.resume_snapshot()
        console.info("Resumed Snapshot")
    except FileNotFoundError:
        if args.pretrained_filename:
            trainer.resume_from_file(args.pretrained_filename)
            console.info(f"Resuming pre-trained model from {args.pretrained_filename}")
        else:
            console.info("Training from scratch")

    trainer.train()

    # End DDP
    end_ddp(trainer_config.enable_ddp)


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seed:
        tmp_args = args
        tmp_args.seed = seed
        tmp_args.out_dir = os.path.join(args.out_dir, f"seed_{seed}")
        set_seed(seed)
        try:
            main(tmp_args)
        except Exception as e:
            logging.critical(e, exc_info=True)
