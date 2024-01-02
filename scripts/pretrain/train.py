import os
import argparse

from torch.distributed.elastic.multiprocessing.errors import record
from hybrid_transformer.utils.loggers.wandb import WandbLogger

from hybrid_transformer.configs.task import TaskConfig
from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.configs.trainer import TrainerConfig
from hybrid_transformer.configs.logger import LoggerConfig

from hybrid_transformer.utils.datasets.auto import AutoDataset
from hybrid_transformer.utils.tokenizers.auto import AutoTokenizer
from hybrid_transformer.models.auto import AutoModel

from hybrid_transformer.trainers.trainer import Trainer


DEFAULT_CONFIG_FILES = {
    'task': "./configs/tasks/guacamol/distribution_learning/",
    'model': "./configs/models/hybrid_transformer/",
    'trainer': "./configs/trainers/pretrain/",
    'logger': "./configs/loggers/wandb/"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("-task", "--path_to_task_config", type=str, default=DEFAULT_CONFIG_FILES['task'])
    parser.add_argument("-model", "--path_to_model_config", type=str, default=DEFAULT_CONFIG_FILES['model'])
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("-logger", "--path_to_logger_config", type=str, default=DEFAULT_CONFIG_FILES['logger'])
    args = parser.parse_args()
    return args


@record
def main():

    # Args
    args = parse_args()

    # Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

    # Override
    trainer_config.out_dir = args.out_dir
    print("Out dir: {}".format(trainer_config.out_dir))

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
    if trainer.resume and os.path.isfile(os.path.join(trainer.out_dir, 'ckpt.pt')):
        trainer.load_checkpoint()
    # Train
    trainer.train()

    # save last
    trainer.save_checkpoint(os.path.join(trainer.out_dir, 'last'))
    logger.finish()


if __name__ == "__main__":
    main()
