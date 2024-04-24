import os
import argparse

from torch.distributed import init_process_group, destroy_process_group

from torch.distributed.elastic.multiprocessing.errors import record
from jointformer.utils.loggers.wandb import WandbLogger

from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel

from jointformer.trainers.trainer import Trainer


DEFAULT_CONFIG_FILES = {
    'task': "./configs/tasks/distribution_learning/guacamol/",
    'model': "./configs/models/gpt_finetune/",
    'trainer': "./configs/trainers/pretrain/",
    'logger': "./configs/loggers/wandb/"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", "--path_to_task_config", type=str, default=DEFAULT_CONFIG_FILES['task'])
    parser.add_argument("-model", "--path_to_model_config", type=str, default=DEFAULT_CONFIG_FILES['model'])
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("-logger", "--path_to_logger_config", type=str, default=DEFAULT_CONFIG_FILES['logger'])
    args = parser.parse_args()
    return args


def ddp_setup():
    init_process_group(backend="nccl")


@record
def main():

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_setup()

    # Args
    args = parse_args()

    # Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

    # Init
    train_dataset = AutoDataset.from_config(task_config, split='train')
    eval_dataset = AutoDataset.from_config(task_config, split='val')
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)
    logger = WandbLogger(logger_config, [task_config, model_config, trainer_config])
    trainer = Trainer(
        config=trainer_config, model=model, train_dataset=train_dataset,
        eval_dataset=eval_dataset, tokenizer=tokenizer, logger=logger)

    # Load
    if trainer.resume and os.path.isfile(os.path.join(trainer.out_dir, 'ckpt')):
        trainer.load_checkpoint()
    # Train
    trainer.train()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
