import os
import argparse
import json

import wandb
from tqdm import tqdm
from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.smiles.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.wandb import WandbLogger

from jointformer.trainers.trainer import Trainer

from scripts.joint_learning.train import DEFAULT_CONFIG_FILES

from jointformer.utils.objectives.guacamol.objective import GUACAMOL_TASKS
from jointformer.models.prediction import PREDICTION_MODEL_CONFIGS
from jointformer.utils.objectives.molecule_net.objective import MOLECULE_NET_REGRESSION_TASKS


from scripts.pretrain.eval import DEFAULT_REFERENCE_FILE, evaluate_distribution_learning


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model", type=str, nargs='?')
    parser.add_argument('--task_p', nargs='?', type=float)
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("-logger", "--path_to_logger_config", type=str, default=DEFAULT_CONFIG_FILES['logger'])
    parser.add_argument("--reference_file", type=str, default=DEFAULT_REFERENCE_FILE)
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

    args.model = args.model if args.model is not None else list(PREDICTION_MODEL_CONFIGS.keys())
    print(f"Evaluating the following models: {args.model} with {args.reference_file}...")

    for task in tqdm(TASKS, desc=f'Evaluating {args.benchmark} benchmark'):
        task_config = TaskConfig.from_pretrained(task_config_path())
        task_config.augmentation_prob = 0.0  # disable augmentation
        dataset = AutoDataset.from_config(task_config, split='test')
        tokenizer = AutoTokenizer.from_config(task_config)

        for model_name, path_to_model_config in tqdm(PREDICTION_MODEL_CONFIGS.items(), desc=f'Evaluating {task}'):

            if model_name in args.model:
                print(f"Evaluating {model_name}...")
                model_config = ModelConfig.from_pretrained(path_to_model_config)

                trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
                logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

                run_dir = f'{model_name}/{task}' if args.task_p is None else f'{model_name}/{task}/{args.task_p}'
                out_dir = os.path.join(args.out_dir, run_dir)
                trainer_config.out_dir = out_dir

                logger_config.project = 'Joint Learning Constant Learning Rate Test ' + args.benchmark
                logger_config.name = model_name + '_' + task if args.task_p is None else model_name + '_' + str(args.task_p) + '_' + task

                model = AutoModel.from_config(model_config)
                logger = WandbLogger(logger_config, [task_config, model_config, trainer_config])
                trainer = Trainer(config=trainer_config, model=model, tokenizer=tokenizer, logger=logger)
                trainer.load_checkpoint()

                # Evaluate Predictive Task
                logger_init = False
                filename = os.path.join(trainer.out_dir, 'results_prediction.json')
                if not os.path.isfile(filename):
                    logger_init = True
                    results_prediction = trainer.test(dataset)

                    with open(filename, 'w') as fp:
                        json.dump(results_prediction, fp)

                # Evaluate Generative Task
                if args.reference_file == DEFAULT_REFERENCE_FILE:
                    filename = os.path.join(trainer.out_dir, "distribution_learning_results.json")
                else:
                    filename = os.path.join(trainer.out_dir, "distribution_learning_results_finetune.json")
                if args.benchmark == 'guacamol' and not os.path.isfile(filename):
                    logger_init = True
                    results = evaluate_distribution_learning(trainer, args.reference_file, filename)
                    try:
                        wandb.log(results)
                    except wandb.Error:
                        pass
                if logger_init:
                    trainer.logger.finish()


if __name__ == "__main__":
    main()
