import os
import json
import argparse

from guacamol.assess_distribution_learning import assess_distribution_learning

from hybrid_transformer.configs.task import TaskConfig
from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.configs.trainer import TrainerConfig
from hybrid_transformer.models.auto import AutoModel
from hybrid_transformer.models.utils import GuacamolModelWrapper
from hybrid_transformer.trainers.trainer import Trainer
from hybrid_transformer.utils.tokenizers.auto import AutoTokenizer

from scripts.joint_learning.train import DEFAULT_CONFIG_FILES

from scripts.pretrain.train import DEFAULT_CONFIG_FILES


DEFAULT_REFERENCE_FILE = '../data/guacamol/train/smiles.txt'
BATCH_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("-task", "--path_to_task_config", type=str, default=DEFAULT_CONFIG_FILES['task'])
    parser.add_argument("-model", "--path_to_model_config", type=str, default=DEFAULT_CONFIG_FILES['model'])
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    parser.add_argument("--reference_file", type=str, default=DEFAULT_REFERENCE_FILE)
    args = parser.parse_args()
    return args


def evaluate_distribution_learning(trainer, reference_file):
    trainer._train_init()
    assess_distribution_learning(
        model=GuacamolModelWrapper(trainer.model, trainer.tokenizer, BATCH_SIZE, trainer.device),
        chembl_training_file=reference_file,
        json_output_file=os.path.join(trainer.out_dir, "distribution_learning_results.json"),
        benchmark_version='v2')

    with open(os.path.join(trainer.out_dir, "distribution_learning_results.json")) as json_data:
        data = json.load(json_data)
    results = {
        "validity": data['results'][0]['score'],
        "uniqueness": data['results'][1]['score'],
        "novelty": data['results'][2]['score'],
        "kl_div": data['results'][3]['score'],
        "fcd": data['results'][4]['score'],
    }
    return results


def main():

    # Args
    args = parse_args()

    # Configs
    task_config = TaskConfig.from_pretrained(args.path_to_task_config)
    model_config = ModelConfig.from_pretrained(args.path_to_model_config)
    trainer_config = TrainerConfig.from_pretrained(args.path_to_trainer_config)

    # Override
    trainer_config.out_dir = args.out_dir
    print("Out dir: {}".format(trainer_config.out_dir))

    # Init
    tokenizer = AutoTokenizer.from_config(task_config)
    model = AutoModel.from_config(model_config)
    trainer = Trainer(config=trainer_config, model=model, tokenizer=tokenizer)

    # Load
    if os.path.isfile(os.path.join(trainer.out_dir, 'ckpt.pt')):
        trainer.load_checkpoint()
    else:
        print("No checkpoint file in {}".format(trainer.out_dir))

    evaluate_distribution_learning(trainer=trainer, reference_file=args.reference_file)


if __name__ == "__main__":
    main()
