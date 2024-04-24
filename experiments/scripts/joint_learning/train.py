import os
import argparse

from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.wandb import WandbLogger

from jointformer.trainers.trainer import Trainer

from jointformer.utils.targets.smiles.guacamol import GUACAMOL_TASKS
from jointformer.utils.targets.smiles.molecule_net import MOLECULE_NET_REGRESSION_TASKS
from jointformer.models.prediction import PREDICTION_MODEL_CONFIGS

DEFAULT_CONFIG_FILES = {
    'trainer': "./configs/trainers/finetune-prediction/",
    'logger': "./configs/loggers/wandb/"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model", type=str, nargs='?')
    parser.add_argument('--task_p', nargs='?', type=float)
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

    else:
        raise ValueError('Provide a correct benchmark name.')

    print("Initializing experiment with tasks {} and models {}".format(TASKS, args.model))

    args.model = args.model if args.model is not None else list(PREDICTION_MODEL_CONFIGS.keys())
    print(f"Training the following models {args.model}")

    for task in TASKS:
        task_config = TaskConfig.from_pretrained(task_config_path())
        train_dataset = AutoDataset.from_config(task_config, split='train')
        eval_dataset = AutoDataset.from_config(task_config, split='val')
        tokenizer = AutoTokenizer.from_config(task_config)

        for model_name, path_to_model_config in PREDICTION_MODEL_CONFIGS.items():

            if model_name in args.model:
                print("Running model {}".format(model_name))
                model_config = ModelConfig.from_pretrained(path_to_model_config)
                trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
                logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

                if model_config.model_name == 'HybridTransformer':
                    # multiplication_factor = 1 / (1 - model_config.task_p)
                    # trainer_config.max_iters = int(trainer_config.max_iters * multiplication_factor)
                    # print("Multiplication factor: ", multiplication_factor)

                    if args.task_p is not None:
                        model_config.task_p = args.task_p

                run_dir = f'{model_name}/{task}' if args.task_p is None else f'{model_name}/{task}/{args.task_p}'
                out_dir = os.path.join(args.out_dir, run_dir)
                print(f"Setting output directory `out_dir` to {out_dir}")
                trainer_config.out_dir = out_dir
                logger_config.project = 'Joint Learning Constant Learning Rate ' + args.benchmark
                logger_config.name = model_name + '_' + task if args.task_p is None else model_name + '_' + str(args.task_p) + '_' + task

                model = AutoModel.from_config(model_config)
                logger = WandbLogger(logger_config, [task_config, model_config, trainer_config])
                trainer = Trainer(
                    config=trainer_config, model=model, train_dataset=train_dataset,
                    eval_dataset=eval_dataset, tokenizer=tokenizer, logger=logger)

                print(f"Loading checkpoint from {model_config.path_to_pretrained}")
                trainer.load_checkpoint(path_to_checkpoint=model_config.path_to_pretrained, reset_iter_num=True)
                print(f"Fine-tuning {model_config.model_name}...")
                trainer.train()
                # print(f"Saving last checkpoint to {trainer.out_dir}/last/ ...")
                # trainer.save_checkpoint(os.path.join(trainer.out_dir, 'last'))
                trainer.logger.finish()


if __name__ == "__main__":
    main()
