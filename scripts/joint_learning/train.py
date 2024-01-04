import os
import argparse
import torch
import wandb

from hybrid_transformer.configs.task import TaskConfig
from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.configs.trainer import TrainerConfig
from hybrid_transformer.configs.logger import LoggerConfig

from hybrid_transformer.utils.datasets.auto import AutoDataset
from hybrid_transformer.utils.tokenizers.auto import AutoTokenizer
from hybrid_transformer.models.auto import AutoModel
from hybrid_transformer.utils.loggers.wandb import WandbLogger

from hybrid_transformer.trainers.trainer import Trainer

from hybrid_transformer.utils.objectives.guacamol.objective import GUACAMOL_TASKS
from hybrid_transformer.utils.objectives.molecule_net.objective import MOLECULE_NET_REGRESSION_TASKS
from hybrid_transformer.models.prediction import PREDICTION_MODEL_CONFIGS

DEFAULT_CONFIG_FILES = {
    'trainer': "./configs/trainers/finetune-prediction/",
    'logger': "./configs/loggers/wandb/"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("-logger", "--path_to_logger_config", type=str, default=DEFAULT_CONFIG_FILES['logger'])
    args = parser.parse_args()
    return args


def main():

    # Args
    args = parse_args()

    # Load configs
    if args.benchmark == 'guacamol':
        task_config_path = lambda: f'./configs/tasks/guacamol/{task}/config.json'
        TASKS = GUACAMOL_TASKS
    elif args.benchmark == 'molecule_net':
        task_config_path = lambda: f'./configs/tasks/molecule_net/{task}/config.json'
        TASKS = MOLECULE_NET_REGRESSION_TASKS
        PREDICTION_MODEL_CONFIGS.pop('GPTForPrediction')
        PREDICTION_MODEL_CONFIGS.pop('JointGPT')
    else:
        raise ValueError('Provide a correct benchmark name.')

    for task in TASKS:
        task_config = TaskConfig.from_pretrained(task_config_path())
        train_dataset = AutoDataset.from_config(task_config, split='train')
        eval_dataset = AutoDataset.from_config(task_config, split='val')
        tokenizer = AutoTokenizer.from_config(task_config)

        for model_name, path_to_model_config in PREDICTION_MODEL_CONFIGS.items():

            model_config = ModelConfig.from_pretrained(path_to_model_config)

            trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
            logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

            run_dir = f'{model_name}/{task}'
            out_dir = os.path.join(args.out_dir, run_dir)
            print(f"Setting output directory `out_dir` to {out_dir}")
            trainer_config.out_dir = out_dir
            logger_config.project = 'Joint Learning'
            logger_config.name = model_name + '_' + task

            model = AutoModel.from_config(model_config)
            logger = WandbLogger(logger_config, [task_config, model_config, trainer_config])
            trainer = Trainer(
                config=trainer_config, model=model, train_dataset=train_dataset,
                eval_dataset=eval_dataset, tokenizer=tokenizer, logger=logger)

            print(f"Loading checkpoint from {model_config.path_to_pretrained}")
            trainer.load_checkpoint(path_to_checkpoint=model_config.path_to_pretrained, reset_iter_num=True)
            print(f"Fine-tuning {model_config.model_name}...")
            trainer.train()
            print(f"Saving last checkpoint to {trainer.out_dir}/last/ ...")
            trainer.save_checkpoint(os.path.join(trainer.out_dir, 'last'))
            trainer.logger.finish()


if __name__ == "__main__":
    main()
