import os
import json
import argparse
import torch

from typing import List
from guacamol.assess_distribution_learning import assess_distribution_learning

from hybrid_transformer.configs.task import TaskConfig
from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.configs.trainer import TrainerConfig
from hybrid_transformer.models.auto import AutoModel
from hybrid_transformer.trainers.trainer import Trainer
from hybrid_transformer.utils.tokenizers.auto import AutoTokenizer

from scripts.joint_learning.train import DEFAULT_CONFIG_FILES

from scripts.pretrain.train import DEFAULT_CONFIG_FILES

FINE_TUNE_REFERENCE_FILE = './data/guacamol/train_10000/smiles.txt'
BATCH_SIZE = 2 #  128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("-task", "--path_to_task_config", type=str, default=DEFAULT_CONFIG_FILES['task'])
    parser.add_argument("-model", "--path_to_model_config", type=str, default=DEFAULT_CONFIG_FILES['model'])
    parser.add_argument("-trainer", "--path_to_trainer_config", type=str, default=DEFAULT_CONFIG_FILES['trainer'])
    args = parser.parse_args()
    return args


@torch.no_grad()
def generate(trainer, num_samples: int) -> List[str]:
    trainer._train_init()
    trainer.model.eval()
    generated_samples = []
    sampling_iters = num_samples // trainer.batch_size + 1
    for _ in range(sampling_iters):
        idx = torch.ones(size=(trainer.batch_size, 1), device=trainer.device) * trainer.tokenizer.generate_token_id
        idx = idx.long()
        samples_batch = trainer.model.generate(idx=idx, max_new_tokens=trainer.tokenizer.max_molecule_length)
        generated_samples.extend(samples_batch)
        print(f"Generated {len(generated_samples)} out of {num_samples} samples...", end='\r')
    generated_samples = generated_samples[:num_samples]
    return trainer.tokenizer.decode(generated_samples)


def save_strings_to_file(strings, filename):
    with open(filename, 'w') as f:
        for s in strings:
            f.write(s + '\n')


def read_strings_from_file(filename):
    with open(filename, 'r') as f:
        strings = f.read().splitlines()
    return strings


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
        generated_samples = generate(trainer=trainer, num_samples=10000)
        save_strings_to_file(generated_samples, os.path.join(trainer.out_dir, 'generated.smi'))
        print(f"Successfully generated {len(generated_samples)} samples in {trainer.out_dir}")

    else:
        raise FileNotFoundError("No checkpoint file in {}".format(trainer.out_dir))


if __name__ == "__main__":
    main()
