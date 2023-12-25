import argparse
import os
import wandb

from torch.distributed.elastic.multiprocessing.errors import record

from hybrid_transformer.configs.task import TaskConfig
from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.configs.trainer import TrainerConfig
from hybrid_transformer.configs.logger import LoggerConfig

from hybrid_transformer.utils.datasets.auto import AutoDataset
from hybrid_transformer.utils.tokenizers.auto import AutoTokenizer
from hybrid_transformer.models.auto import AutoModel

from hybrid_transformer.trainers.trainer import Trainer

local_rank = int(os.environ["LOCAL_RANK"])

DEFAULT_CONFIGS = {
    'task': "./configs/tasks/distribution_learning/guacamol/config.json",
    'model': "./configs/models/gpt/config.json",
    'trainers': "./configs/trainers/config.json",
    'logger': "./configs/loggers/wandb/config.json"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", "--path_to_task_config", type=str, default=DEFAULT_CONFIGS['task'])
    parser.add_argument("-model", "--path_to_task_config", type=str, default=DEFAULT_CONFIGS['model'])
    parser.add_argument("-trainer", "--path_to_task_config", type=str, default=DEFAULT_CONFIGS['trainers'])
    parser.add_argument("-logger", "--path_to_logger_config", type=str, default=DEFAULT_CONFIGS['logger'])
    parser.add_argument("-d", "--debug", type=bool, default=False)
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

    # Init
    train_dataset = AutoDataset.from_config(task_config, split='train')
    eval_dataset = AutoDataset.from_config(task_config, split='val')
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)
    trainer = Trainer(
        config=trainer_config, model=model, train_dataset=train_dataset,
        eval_dataset=eval_dataset, tokenizer=tokenizer, wandb_log=logger_config.wandb_log)

    if logger_config.wandb_log and trainer.master_process:
        run = wandb.init(
            project=logger_config.wandb_project,
            name=logger_config.run_name,
            resume=True,
            config={
                'task_config': task_config,
                'model_config': model_config,
                'trainer_config': trainer_config,
                'logger_config': logger_config
            }
        )
    trainer.train()


if __name__ == "__main__":
    main()
