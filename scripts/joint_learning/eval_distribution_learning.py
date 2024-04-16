import os
import argparse

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


from scripts.pretrain.eval import DEFAULT_REFERENCE_FILE, evaluate_distribution_learning

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("-logger", "--path_to_logger_config", type=str, default=DEFAULT_CONFIG_FILES['logger'])
    parser.add_argument("--reference_file", type=str, default=DEFAULT_REFERENCE_FILE)
    args = parser.parse_args()
    return args


def main():

    # Args
    args = parse_args()

    # Load configs
    task_config_path = lambda: f'./configs/tasks/guacamol/{guacamol_task}/config.json'

    for guacamol_task in tqdm(GUACAMOL_TASKS):
        task_config = TaskConfig.from_pretrained(task_config_path())
        task_config.augmentation_prob = 0.0  # disable augmentation
        dataset = AutoDataset.from_config(task_config, split='test')
        tokenizer = AutoTokenizer.from_config(task_config)

        for model_name, path_to_model_config in tqdm(PREDICTION_MODEL_CONFIGS.items()):

            model_config = ModelConfig.from_pretrained(path_to_model_config)

            trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
            logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

            run_dir = f'{model_name}/{guacamol_task}'
            out_dir = os.path.join(args.out_dir, run_dir)
            trainer_config.out_dir = out_dir

            logger_config.project = 'Table 1 Test'
            logger_config.name = model_name + '_' + guacamol_task

            model = AutoModel.from_config(model_config)
            logger = WandbLogger(logger_config, [task_config, model_config, trainer_config])
            trainer = Trainer(config=trainer_config, model=model, tokenizer=tokenizer, logger=logger)

            print(f"Loading checkpoint from {trainer.out_dir}...")
            trainer.load_checkpoint()
            # prediction_results = trainer.test(dataset)

            # with open(os.path.join(trainer.out_dir, 'results_prediction.json'), 'w') as fp:
            #     json.dump(prediction_results, fp)

            results = evaluate_distribution_learning(trainer, args.reference_file)

            if logger is not None:
                wandb.log(results)

            trainer.logger.finish()


if __name__ == "__main__":
    main()
