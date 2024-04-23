import os
import argparse
import json

from tqdm import tqdm
from jointformer.configs.task import TaskConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.wandb import WandbLogger

from jointformer.trainers.trainer import Trainer

from scripts.joint_learning.train import DEFAULT_CONFIG_FILES

from jointformer.utils.objectives.guacamol.objective import GUACAMOL_TASKS

from jointformer.utils.datasets.smiles.guacamol import GuacamolSMILESDataset

from experiments.scripts.pretrain.eval import DEFAULT_REFERENCE_FILE

from jointformer.utils.datasets.utils import save_list_into_txt

PRETRAINED_MODEL_RESULTS_DIR = '/raid/aizd/jointformer/results/pretrain/gpt'

MODEL_CONFIGS = {
        'GPTForPrediction': './configs/models/prediction/gpt_finetune/config.json',
        'HybridTransformer': './configs/models/prediction/jointformer/config.json'
    }

RESULTS_FILENAME = 'distribution_learning_results.json'
SAMPLED_DATA_DIR = 'sampled'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("-logger", "--path_to_logger_config", type=str, default=DEFAULT_CONFIG_FILES['logger'])
    parser.add_argument("--reference_file", type=str, default=DEFAULT_REFERENCE_FILE)
    args = parser.parse_args()
    return args


def main():

    # Args
    args = parse_args()

    # Load configs
    task_config_path = lambda: f'./configs/tasks/guacamol/{task}/config.json'
    TASKS = GUACAMOL_TASKS

    for model_name, path_to_model_config in tqdm(MODEL_CONFIGS.items()):

        for task in tqdm(TASKS):

            task_config = TaskConfig.from_pretrained(task_config_path())
            model_config = ModelConfig.from_pretrained(path_to_model_config)
            trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)
            logger_config = LoggerConfig.from_pretrained(args.path_to_logger_config)

            run_dir = f'{model_name}/{task}'
            if model_name == 'GPTForPrediction':
                out_dir = os.path.join('/raid/aizd/jointformer/results_v4/joint_learning/guacamol', run_dir)
            elif model_name == 'HybridTransformer':
                out_dir = os.path.join('/raid/aizd/jointformer/results_v3/joint_learning/guacamol', run_dir)
            else:
                raise NotImplementedError
            trainer_config.out_dir = out_dir

            logger_config.project = 'Generalization'
            logger_config.name = model_name + '_' + task

            tokenizer = AutoTokenizer.from_config(task_config)
            model = AutoModel.from_config(model_config)
            logger = WandbLogger(logger_config, [task_config, model_config, trainer_config])
            trainer = Trainer(config=trainer_config, model=model, tokenizer=tokenizer, logger=logger)
            trainer.load_checkpoint()

            # extract generated data from results
            if model_name == 'GPTForPrediction':
                samples_filename = os.path.join(PRETRAINED_MODEL_RESULTS_DIR, RESULTS_FILENAME)
            elif model_name == 'HybridTransformer':
                samples_filename = os.path.join(trainer.out_dir, RESULTS_FILENAME)
            else:
                raise NotImplementedError

            sampled_data_dir = os.path.join(trainer.out_dir, SAMPLED_DATA_DIR)

            with open(samples_filename) as json_data:
                samples = json.load(json_data)['samples']

            if not os.path.isdir(sampled_data_dir):
                os.makedirs(sampled_data_dir)
            save_list_into_txt(os.path.join(sampled_data_dir, 'smiles_tokenizers.txt'), samples)

            # load data
            dataset = GuacamolSMILESDataset(target_label=task_config.target_label, data_dir=sampled_data_dir)

            # Evaluate Predictive Task
            logger_init = False
            filename = os.path.join(trainer.out_dir, 'results_generalization.json')
            if not os.path.isfile(filename):
                logger_init = True
                results_prediction = trainer.test(dataset)

                with open(filename, 'w') as fp:
                    json.dump(results_prediction, fp)

            if logger_init:
                trainer.logger.finish()


if __name__ == "__main__":
    main()
