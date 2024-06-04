import os
import logging
import argparse
import time

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

from jointformer.utils.runtime import set_seed, log_args, create_output_dir

process_timestamp = time.strftime("%Y%m%d-%H%M%S")
console_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename="process.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logging.captureWarnings(True)

DEFAULT_SEED_ARRAY = [1337]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    log_args(args)
    return args


@record
def main(args):

    # Load Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)


    # Create output directory
    create_output_dir(args.out_dir)

    # Load data, tokenizer and model
    train_dataset = AutoDataset.from_config(task_config, split='train')
    val_dataset = AutoDataset.from_config(task_config, split='val')
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config)
    logger.store_configs(task_config, model_config, trainer_config, logger_config)
    logger.save_configs(args.out_dir)
    if args.logger_display_name is not None:
        logger.set_display_name(args.logger_display_name)

    trainer = Trainer(
        out_dir=args.out_dir,
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
        console_logger.info("Resumed Snapshot")
    except FileNotFoundError:
        if args.pretrained_filename:
            trainer.resume_from_file(args.pretrained_filename)
            console_logger.info(f"Resuming pre-trained model from {args.pretrained_filename}")
        else:
            console_logger.info("Training from scratch")

    trainer.train()


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
