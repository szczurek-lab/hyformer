import argparse
import torch

from configs.data import DataConfig
from configs.models import ModelConfig
from configs.molecule_generation import MoleculeGenerationConfig

from joint_transformer.tokenizers.tokenizer import SmilesTokenizer
from joint_transformer.data.loader import get_loader_unsupervised
from joint_transformer.trainers.pre_trainer import PreTrainer

from joint_transformer.utils.set_seed import set_seed

CONFIG_FILES = {
    'data': './configs/data/guacamol_config.json',
    'models': '.configs/models/hybrid_transformer.json',
    'trainer': '.configs/trainer/distribution_learning_trainer.json',
    'experiments': '.configs/experiments/distribution_learning/guacamol/configs.json',
}

def parse_args() -> argparse.Namespace:
    # All arguments can be modified here
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='transformer')
    parser.add_argument('--size', type=str, default='small')
    parser.add_argument('--task_p', type=float, default=0.95)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    config = MoleculeGenerationConfig(
        data_config=DataConfig(),
        model_config=ModelConfig(model_type=args.model_type, size=args.size, task_p=args.task_p))
    config.task_p = args.task_p
    device = torch.device(config.device)
    set_seed(config.seed)
    tokenizer = SmilesTokenizer(config.data_config.path_vocab,
                                max_molecule_length=config.data_config.max_molecule_length)

    loader_train = get_loader_unsupervised(config.data_config.path_data_train,
                                           tokenizer, config.batch_size, device)
    loader_valid = get_loader_unsupervised(config.data_config.path_data_valid,
                                           tokenizer, config.batch_size, device)

    model = config.model_config.model_class(
        tokenizer=tokenizer,
        num_layers=config.model_config.num_layers,
        embed_dim=config.model_config.embed_dim,
        num_heads=config.model_config.num_heads,
        dim_feedforward=config.model_config.dim_feedforward,
        dropout=config.model_config.dropout,
        bias=config.model_config.bias,
        flash=config.model_config.flash,
        verbose=True)
    model.to(device)

    trainer = PreTrainer(
        loader_train=loader_train, loader_valid=loader_valid, tokenizer=tokenizer, model=model, config=config,
        device=device)
    trainer.train()


if __name__ == '__main__':
    main()